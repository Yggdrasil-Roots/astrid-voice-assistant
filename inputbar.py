"""The text-input bar for Astrid's window: typing, file attachments, and a switch
for spoken replies to typed messages.

Kept apart from gui.py so it can be tested without the models, the microphone or
the rest of the window. It owns only presentation and the list of chosen file
paths; everything that decides what to DO with a message lives in the window,
which hands the bar plain callbacks. All methods must be called on the GTK main
thread.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Sequence

import gi

gi.require_version("Gtk", "4.0")
gi.require_version("Gdk", "4.0")
from gi.repository import Gdk, Gio, GLib, Gtk  # noqa: E402

MAX_ATTACHMENTS = 5
STATUS_SECONDS = 6
CHIP_NAME_CHARS = 26

CSS = """
.input-bar { margin-top: 8px; }
.input-scroller {
    background-color: #0d1015;
    border: 1px solid #262b33;
    border-radius: 6px;
}
.input-scroller:focus-within { border-color: #665221; }
.input-text, .input-text text {
    background-color: #0d1015;
    color: #e6ddcc;
    font-family: monospace;
}
.input-chip {
    background-color: #151a21;
    border: 1px solid #665221;
    border-radius: 12px;
    padding: 1px 4px 1px 10px;
}
.input-chip label { color: #d4ad40; font-size: 12px; }
.input-chip button { min-height: 18px; min-width: 18px; padding: 0; }
.input-status { color: #8c857a; font-size: 12px; }
.input-status.error { color: #e05a4f; }
.input-speak label { color: #8c857a; font-size: 11px; letter-spacing: 1px; }
"""


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024
    return f"{num_bytes} B"


def short_name(path: str, limit: int = CHIP_NAME_CHARS) -> str:
    name = os.path.basename(path) or path
    return name if len(name) <= limit else name[: limit - 1] + "…"


class InputBar(Gtk.Box):
    """A multi-line text box with an attach button and a Send/Stop button.

    Enter sends, Shift+Enter inserts a newline (so multi-line text pastes and
    types naturally), Escape stops a running turn, Ctrl+O opens the file chooser.
    """

    def __init__(
        self,
        *,
        on_submit: Callable[[str, list[str]], None],
        on_stop: Callable[[], None],
        on_attach_paths: Callable[[Sequence[str]], None],
        on_speak_toggled: Callable[[bool], None],
        speak_active: bool = False,
    ) -> None:
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=6)
        self.add_css_class("input-bar")
        self._on_submit = on_submit
        self._on_stop = on_stop
        self._on_attach_paths = on_attach_paths
        self._on_speak_toggled = on_speak_toggled
        self._paths: list[str] = []
        self._chips: dict[str, Gtk.Widget] = {}
        self._enabled = False
        self._busy = False
        self._status_timer = 0

        self._status = Gtk.Label(xalign=0.0, wrap=True)
        self._status.add_css_class("input-status")
        self._status.set_visible(False)
        self.append(self._status)

        self._chip_box = Gtk.FlowBox(selection_mode=Gtk.SelectionMode.NONE,
                                     max_children_per_line=5, homogeneous=False,
                                     column_spacing=6, row_spacing=4)
        self._chip_box.set_visible(False)
        self.append(self._chip_box)

        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        self._attach_button = Gtk.Button.new_from_icon_name("mail-attachment-symbolic")
        self._attach_button.set_tooltip_text("Attach files (Ctrl+O), or drop them on the window")
        self._attach_button.set_valign(Gtk.Align.END)
        self._attach_button.connect("clicked", lambda *_: self.open_file_dialog())
        row.append(self._attach_button)

        self._text = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR, accepts_tab=False,
                                  top_margin=6, bottom_margin=6, left_margin=8, right_margin=8)
        self._text.add_css_class("input-text")
        self._buffer = self._text.get_buffer()
        scroller = Gtk.ScrolledWindow(hexpand=True, propagate_natural_height=True,
                                      min_content_height=40, max_content_height=130)
        scroller.add_css_class("input-scroller")
        scroller.set_child(self._text)
        row.append(scroller)

        self._send_button = Gtk.Button.new_from_icon_name("mail-send-symbolic")
        self._send_button.add_css_class("suggested-action")
        self._send_button.set_valign(Gtk.Align.END)
        self._send_button.connect("clicked", self._on_send_clicked)
        row.append(self._send_button)
        self.append(row)

        speak_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8,
                            halign=Gtk.Align.END)
        speak_row.add_css_class("input-speak")
        speak_row.append(Gtk.Label(label="SPEAK REPLIES TO TYPED MESSAGES"))
        self._speak_switch = Gtk.Switch(active=speak_active, valign=Gtk.Align.CENTER)
        self._speak_switch.connect("notify::active", self._on_switch)
        speak_row.append(self._speak_switch)
        self.append(speak_row)

        key = Gtk.EventControllerKey()
        key.connect("key-pressed", self._on_key)
        self._text.add_controller(key)
        self._text.connect("paste-clipboard", self._on_paste)
        self._apply_sensitivity()

    # ------------------------------------------------------------------- state
    @property
    def attachments(self) -> list[str]:
        return list(self._paths)

    @property
    def speak_active(self) -> bool:
        return self._speak_switch.get_active()

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def enabled(self) -> bool:
        return self._enabled

    def get_text(self) -> str:
        return self._buffer.get_text(self._buffer.get_start_iter(),
                                     self._buffer.get_end_iter(), False)

    def set_text(self, text: str) -> None:
        self._buffer.set_text(text)

    def clear(self) -> None:
        """Empty the text box and drop every attachment."""
        self._buffer.set_text("")
        for path in list(self._paths):
            self.remove_attachment(path)

    def set_enabled(self, enabled: bool) -> None:
        """Whether a new message may be sent right now (models ready, screen unlocked)."""
        self._enabled = bool(enabled)
        self._apply_sensitivity()

    def set_busy(self, busy: bool) -> None:
        """While busy the Send button becomes Stop."""
        self._busy = bool(busy)
        self._send_button.set_icon_name("process-stop-symbolic" if self._busy
                                        else "mail-send-symbolic")
        self._send_button.set_tooltip_text("Stop (Esc)" if self._busy else "Send (Enter)")
        self._apply_sensitivity()

    def _apply_sensitivity(self) -> None:
        self._send_button.set_tooltip_text("Stop (Esc)" if self._busy else "Send (Enter)")
        self._send_button.set_sensitive(self._busy or self._enabled)
        self._attach_button.set_sensitive(self._enabled and not self._busy)
        self._text.set_editable(True)

    # ------------------------------------------------------------- attachments
    def add_attachment(self, path: str) -> bool:
        """Add a file chip. Returns False if it was a duplicate or the limit is hit."""
        if path in self._paths:
            return False
        if len(self._paths) >= MAX_ATTACHMENTS:
            self.show_status(f"At most {MAX_ATTACHMENTS} files per message.", error=True)
            return False
        self._paths.append(path)
        chip = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        chip.add_css_class("input-chip")
        try:
            size = human_size(os.stat(path).st_size)
            label = Gtk.Label(label=f"{short_name(path)} · {size}")
        except OSError:
            label = Gtk.Label(label=short_name(path))
        label.set_tooltip_text(path)
        chip.append(label)
        close = Gtk.Button.new_from_icon_name("window-close-symbolic")
        close.add_css_class("flat")
        close.set_tooltip_text("Remove")
        close.connect("clicked", lambda *_: self.remove_attachment(path))
        chip.append(close)
        self._chip_box.append(chip)
        self._chips[path] = chip
        self._chip_box.set_visible(True)
        return True

    def remove_attachment(self, path: str) -> None:
        if path not in self._paths:
            return
        self._paths.remove(path)
        chip = self._chips.pop(path, None)
        if chip is not None:
            parent = chip.get_parent()          # the FlowBoxChild wrapper
            if parent is not None:
                self._chip_box.remove(parent)
        self._chip_box.set_visible(bool(self._paths))

    def open_file_dialog(self) -> None:
        if not (self._enabled and not self._busy):
            return
        dialog = Gtk.FileDialog()
        dialog.set_title("Attach files")
        dialog.set_modal(True)
        dialog.set_initial_folder(Gio.File.new_for_path(os.path.expanduser("~")))
        dialog.open_multiple(self.get_root(), None, self._on_files_chosen)

    def _on_files_chosen(self, dialog: Gtk.FileDialog, result: Gio.AsyncResult) -> None:
        try:
            chosen = dialog.open_multiple_finish(result)
        except GLib.Error:
            return                              # dismissed
        paths = [f.get_path() for f in chosen if f.get_path()]
        if paths:
            self._on_attach_paths(paths)

    def make_drop_target(self) -> Gtk.DropTarget:
        """A drop target for files. The window adds it to itself so a drop works
        anywhere, not just over the bar."""
        target = Gtk.DropTarget.new(Gdk.FileList, Gdk.DragAction.COPY)
        target.connect("drop", self._on_drop)
        return target

    def _on_drop(self, _target: Gtk.DropTarget, value: Gdk.FileList, _x: float, _y: float) -> bool:
        paths = [f.get_path() for f in value.get_files() if f.get_path()]
        if not paths or not self._enabled or self._busy:
            return False
        self._on_attach_paths(paths)
        return True

    def _on_paste(self, text_view: Gtk.TextView) -> None:
        """Ctrl+V: files copied from a file manager become attachments; anything
        else pastes as text through the normal path."""
        clipboard = text_view.get_clipboard()
        if clipboard.get_formats().contain_gtype(Gdk.FileList):
            clipboard.read_value_async(Gdk.FileList, GLib.PRIORITY_DEFAULT, None,
                                       self._on_clipboard_files)
            text_view.stop_emission_by_name("paste-clipboard")

    def _on_clipboard_files(self, clipboard: Gdk.Clipboard, result: Gio.AsyncResult) -> None:
        try:
            value = clipboard.read_value_finish(result)
        except GLib.Error:
            return
        paths = [f.get_path() for f in value.get_files() if f.get_path()]
        if paths and self._enabled and not self._busy:
            self._on_attach_paths(paths)

    # ------------------------------------------------------------------- input
    def _on_key(self, _ctrl: Gtk.EventControllerKey, keyval: int, _keycode: int,
                state: Gdk.ModifierType) -> bool:
        if keyval in (Gdk.KEY_Return, Gdk.KEY_KP_Enter):
            if state & Gdk.ModifierType.SHIFT_MASK:
                return False                    # newline
            self.submit()
            return True
        if keyval == Gdk.KEY_Escape and self._busy:
            self._on_stop()
            return True
        if keyval in (Gdk.KEY_o, Gdk.KEY_O) and state & Gdk.ModifierType.CONTROL_MASK:
            self.open_file_dialog()
            return True
        return False

    def _on_send_clicked(self, _button: Gtk.Button) -> None:
        if self._busy:
            self._on_stop()
        else:
            self.submit()

    def submit(self) -> bool:
        """Send what is in the box. Returns True if the message was handed on."""
        if self._busy:
            self.show_status("She is busy. Press Stop, or wait a moment.")
            return False
        if not self._enabled:
            self.show_status("Not ready yet.")
            return False
        text = self.get_text()
        if not text.strip() and not self._paths:
            return False
        self._on_submit(text, list(self._paths))
        return True

    def _on_switch(self, switch: Gtk.Switch, _param) -> None:
        self._on_speak_toggled(switch.get_active())

    # ------------------------------------------------------------------ status
    def show_status(self, message: str, *, error: bool = False) -> None:
        """A short message above the box that clears itself."""
        self._status.set_text(message)
        if error:
            self._status.add_css_class("error")
        else:
            self._status.remove_css_class("error")
        self._status.set_visible(True)
        if self._status_timer:
            GLib.source_remove(self._status_timer)
        self._status_timer = GLib.timeout_add_seconds(STATUS_SECONDS, self._clear_status)

    def _clear_status(self) -> bool:
        self._status.set_visible(False)
        self._status_timer = 0
        return False
