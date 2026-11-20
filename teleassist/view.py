"""Camera image panels for TeleAssist (Kit UI, with OpenCV fallback)."""

from __future__ import annotations

import numpy as np

TABLE_LIVE_TITLE = "Table Camera"
TABLE_OVERLAY_TITLE = "Table Camera Scan"
_MAIN_VIEWPORT_NAMES = ("Viewport", "/OmniverseKit/Viewport", "Viewport Next")


def compose_overlay(
    rgb: np.ndarray,
    boxes: list[tuple[float, float, float, float]] | None = None,
    points_xy: np.ndarray | None = None,
) -> np.ndarray:
    """RGB with optional boxes and grasp pixels drawn on top."""
    image = np.ascontiguousarray(rgb[..., :3])
    if image.dtype != np.uint8:
        if np.nanmax(image) <= 1.0:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.copy()

    import cv2

    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if points_xy is not None and len(points_xy) > 0:
        for x, y in np.asarray(points_xy, dtype=np.float64).reshape(-1, 2):
            cv2.circle(image, (int(round(x)), int(round(y))), 3, (255, 80, 0), -1)
    return image


def _to_rgba_u8(rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    image = np.ascontiguousarray(rgb[..., :3])
    if image.dtype != np.uint8:
        if np.nanmax(image) <= 1.0:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    height, width = image.shape[:2]
    opaque = np.full((height, width, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([image, opaque], axis=2)
    return np.ascontiguousarray(rgba), width, height


class TableCameraWindows:
    """Live table-cam + scan-overlay panels.

    Prefer Kit ``omni.ui`` image windows (same pattern as Isaac Lab streaming
    view). Fall back to OpenCV named windows if Kit UI is unavailable.
    """

    def __init__(
        self,
        resolution: tuple[int, int] = (480, 480),
        *,
        table_camera_prim: str | None = None,
    ) -> None:
        del table_camera_prim  # kept for call-site compatibility; unused
        self._width, self._height = resolution
        self._backend = "none"
        self._live_provider = None
        self._overlay_provider = None
        self._live_window = None
        self._overlay_window = None

        try:
            self._open_kit_windows()
            self._backend = "kit"
            print(
                f"[TeleAssist] Kit panels: '{TABLE_LIVE_TITLE}' + '{TABLE_OVERLAY_TITLE}'",
                flush=True,
            )
        except Exception as error:
            print(f"[TeleAssist] Kit camera panels failed ({error}); using OpenCV", flush=True)
            self._open_opencv_windows()
            self._backend = "opencv"

    def _open_kit_windows(self) -> None:
        import omni.ui as ui

        width, height = self._width, self._height
        self._live_provider = ui.ByteImageProvider()
        self._overlay_provider = ui.ByteImageProvider()

        self._live_window = ui.Window(
            TABLE_LIVE_TITLE,
            width=width + 16,
            height=height + 40,
            visible=True,
            dock_preference=ui.DockPreference.DISABLED,
        )
        self._live_window.position_x = 40
        self._live_window.position_y = 80
        with self._live_window.frame:
            ui.ImageWithProvider(self._live_provider, fill_policy=ui.IwpFillPolicy.IWP_STRETCH)

        self._overlay_window = ui.Window(
            TABLE_OVERLAY_TITLE,
            width=width + 16,
            height=height + 40,
            visible=True,
            dock_preference=ui.DockPreference.DISABLED,
        )
        self._overlay_window.position_x = 40
        self._overlay_window.position_y = 80 + height + 60
        with self._overlay_window.frame:
            ui.ImageWithProvider(self._overlay_provider, fill_policy=ui.IwpFillPolicy.IWP_STRETCH)

        # Placeholder so the panels are not empty before the first camera read.
        blank = np.zeros((height, width, 3), dtype=np.uint8)
        self._set_kit_image(self._live_provider, blank)
        self._set_kit_image(self._overlay_provider, blank)

        self._live_window.visible = True
        self._overlay_window.visible = True
        _dock_camera_panels(self._live_window, self._overlay_window)

    def _open_opencv_windows(self) -> None:
        import cv2

        cv2.namedWindow(TABLE_LIVE_TITLE, cv2.WINDOW_NORMAL)
        cv2.namedWindow(TABLE_OVERLAY_TITLE, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(TABLE_LIVE_TITLE, self._width, self._height)
        cv2.resizeWindow(TABLE_OVERLAY_TITLE, self._width, self._height)

    @staticmethod
    def _set_kit_image(provider, rgb: np.ndarray) -> None:
        rgba, width, height = _to_rgba_u8(rgb)
        # Match Isaac Lab KitVisualizer / image_plot upload path.
        try:
            provider.set_bytes_data(rgba.flatten().data, [width, height])
        except Exception:
            provider.set_data_array(rgba, [width, height])

    def update(
        self,
        rgb: np.ndarray,
        boxes: list[tuple[float, float, float, float]] | None = None,
        points_xy: np.ndarray | None = None,
    ) -> None:
        if rgb is None or np.asarray(rgb).size == 0:
            return
        live = np.ascontiguousarray(rgb[..., :3])
        if live.dtype != np.uint8:
            if np.nanmax(live) <= 1.0:
                live = np.clip(live * 255.0, 0, 255).astype(np.uint8)
            else:
                live = np.clip(live, 0, 255).astype(np.uint8)
        overlay = compose_overlay(rgb, boxes, points_xy)

        if self._backend == "kit":
            self._set_kit_image(self._live_provider, live)
            self._set_kit_image(self._overlay_provider, overlay)
            if self._live_window is not None:
                self._live_window.visible = True
            if self._overlay_window is not None:
                self._overlay_window.visible = True
            return

        if self._backend == "opencv":
            import cv2

            # OpenCV expects BGR for color display.
            cv2.imshow(TABLE_LIVE_TITLE, cv2.cvtColor(live, cv2.COLOR_RGB2BGR))
            cv2.imshow(TABLE_OVERLAY_TITLE, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


def _find_main_viewport(ui) -> object | None:
    for name in _MAIN_VIEWPORT_NAMES:
        window = ui.Workspace.get_window(name)
        if window is not None:
            return window
    # Last resort: any window whose title contains "Viewport".
    try:
        for window in ui.Workspace.get_windows():
            title = getattr(window, "title", None) or getattr(window, "name", "") or ""
            if "viewport" in str(title).lower():
                return window
    except Exception:
        pass
    return None


def _dock_camera_panels(live_window, overlay_window) -> None:
    """Pin both panels into a column on the right of the main scene view."""
    import asyncio

    import omni.kit.app
    import omni.ui as ui

    async def pin() -> None:
        app = omni.kit.app.get_app()
        scene_view = None
        for _ in range(30):
            await app.next_update_async()
            scene_view = _find_main_viewport(ui)
            if scene_view is not None:
                break

        live_window.visible = True
        overlay_window.visible = True
        if scene_view is None:
            print(
                "[TeleAssist] main Viewport not found; leaving camera panels floating",
                flush=True,
            )
            try:
                live_window.focus()
                overlay_window.focus()
            except Exception:
                pass
            return

        try:
            live_window.dock_in(scene_view, ui.DockPosition.RIGHT, 0.30)
            await app.next_update_async()
            overlay_window.dock_in(live_window, ui.DockPosition.BOTTOM, 0.5)
            await app.next_update_async()
            live_window.visible = True
            overlay_window.visible = True
            live_window.focus()
            print(
                f"[TeleAssist] docked camera panels to the right of '{getattr(scene_view, 'title', scene_view)}'",
                flush=True,
            )
        except Exception as error:
            print(f"[TeleAssist] dock failed ({error}); panels should still be visible", flush=True)

    asyncio.ensure_future(pin())
