"""Camera image panels for TeleAssist (Kit UI, with OpenCV fallback)."""

from __future__ import annotations

import numpy as np

TABLE_LIVE_TITLE = "Table Camera"
TABLE_HEATMAP_TITLE = "Affordance Heatmap"
TABLE_OVERLAY_TITLE = "Table Camera Scan"
_MAIN_VIEWPORT_NAMES = ("Viewport", "/OmniverseKit/Viewport", "Viewport Next")


def compose_overlay(
    rgb: np.ndarray,
    boxes: list[tuple[float, float, float, float]] | None = None,
    points_xy: np.ndarray | None = None,
) -> np.ndarray:
    """RGB with optional boxes and grasp pixels drawn on top."""
    image = _as_uint8_rgb(rgb)
    import cv2

    if boxes:
        for box in boxes:
            x1, y1, x2, y2 = (int(round(v)) for v in box)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    if points_xy is not None and len(points_xy) > 0:
        for x, y in np.asarray(points_xy, dtype=np.float64).reshape(-1, 2):
            cv2.circle(image, (int(round(x)), int(round(y))), 3, (255, 80, 0), -1)
    return image


def compose_heatmap_overlay(
    rgb: np.ndarray,
    heatmap: np.ndarray | None,
    boxes: list[tuple[float, float, float, float]] | None = None,
    points_xy: np.ndarray | None = None,
    alpha: float = 0.55,
) -> np.ndarray:
    """RGB blended with a JET affordance heatmap (plus optional boxes/points)."""
    image = _as_uint8_rgb(rgb)
    if heatmap is None or np.asarray(heatmap).size == 0:
        return compose_overlay(image, boxes=boxes, points_xy=points_xy)

    import cv2

    heat = np.asarray(heatmap, dtype=np.float32)
    if heat.shape[:2] != image.shape[:2]:
        heat = cv2.resize(heat, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)
    peak = float(np.nanmax(heat))
    if not np.isfinite(peak) or peak <= 0:
        return compose_overlay(image, boxes=boxes, points_xy=points_xy)

    norm = np.clip(heat / peak, 0.0, 1.0)
    color = cv2.applyColorMap((norm * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
    color = cv2.cvtColor(color, cv2.COLOR_BGR2RGB)
    blended = (
        (1.0 - alpha) * image.astype(np.float32) + alpha * color.astype(np.float32)
    ).astype(np.uint8)
    return compose_overlay(blended, boxes=boxes, points_xy=points_xy)


def _as_uint8_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.ascontiguousarray(rgb[..., :3])
    if image.dtype != np.uint8:
        if np.nanmax(image) <= 1.0:
            image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
        else:
            image = np.clip(image, 0, 255).astype(np.uint8)
    else:
        image = image.copy()
    return image


def _to_rgba_u8(rgb: np.ndarray) -> tuple[np.ndarray, int, int]:
    image = _as_uint8_rgb(rgb)
    height, width = image.shape[:2]
    opaque = np.full((height, width, 1), 255, dtype=np.uint8)
    rgba = np.concatenate([image, opaque], axis=2)
    return np.ascontiguousarray(rgba), width, height


class TableCameraWindows:
    """Live table-cam, affordance heatmap, and scan-overlay panels."""

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
        self._heatmap_provider = None
        self._overlay_provider = None
        self._live_window = None
        self._heatmap_window = None
        self._overlay_window = None

        try:
            self._open_kit_windows()
            self._backend = "kit"
            print(
                f"[TeleAssist] Kit panels: '{TABLE_LIVE_TITLE}', "
                f"'{TABLE_HEATMAP_TITLE}', '{TABLE_OVERLAY_TITLE}'",
                flush=True,
            )
        except Exception as error:
            print(f"[TeleAssist] Kit camera panels failed ({error}); using OpenCV", flush=True)
            self._open_opencv_windows()
            self._backend = "opencv"

    def _make_kit_window(self, ui, title: str, provider, y: int):
        width, height = self._width, self._height
        window = ui.Window(
            title,
            width=width + 16,
            height=height + 40,
            visible=True,
            dock_preference=ui.DockPreference.DISABLED,
        )
        window.position_x = 40
        window.position_y = y
        with window.frame:
            ui.ImageWithProvider(provider, fill_policy=ui.IwpFillPolicy.IWP_STRETCH)
        return window

    def _open_kit_windows(self) -> None:
        import omni.ui as ui

        width, height = self._width, self._height
        self._live_provider = ui.ByteImageProvider()
        self._heatmap_provider = ui.ByteImageProvider()
        self._overlay_provider = ui.ByteImageProvider()

        self._live_window = self._make_kit_window(ui, TABLE_LIVE_TITLE, self._live_provider, 80)
        self._heatmap_window = self._make_kit_window(
            ui, TABLE_HEATMAP_TITLE, self._heatmap_provider, 80 + height + 60
        )
        self._overlay_window = self._make_kit_window(
            ui, TABLE_OVERLAY_TITLE, self._overlay_provider, 80 + 2 * (height + 60)
        )

        blank = np.zeros((height, width, 3), dtype=np.uint8)
        self._set_kit_image(self._live_provider, blank)
        self._set_kit_image(self._heatmap_provider, blank)
        self._set_kit_image(self._overlay_provider, blank)

        for window in (self._live_window, self._heatmap_window, self._overlay_window):
            window.visible = True
        _dock_camera_panels(self._live_window, self._heatmap_window, self._overlay_window)

    def _open_opencv_windows(self) -> None:
        import cv2

        for title in (TABLE_LIVE_TITLE, TABLE_HEATMAP_TITLE, TABLE_OVERLAY_TITLE):
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, self._width, self._height)

    @staticmethod
    def _set_kit_image(provider, rgb: np.ndarray) -> None:
        rgba, width, height = _to_rgba_u8(rgb)
        try:
            provider.set_bytes_data(rgba.flatten().data, [width, height])
        except Exception:
            provider.set_data_array(rgba, [width, height])

    def update(
        self,
        rgb: np.ndarray,
        boxes: list[tuple[float, float, float, float]] | None = None,
        points_xy: np.ndarray | None = None,
        heatmap: np.ndarray | None = None,
    ) -> None:
        if rgb is None or np.asarray(rgb).size == 0:
            return
        live = _as_uint8_rgb(rgb)
        heat_view = compose_heatmap_overlay(rgb, heatmap, boxes=boxes, points_xy=points_xy)
        overlay = compose_overlay(rgb, boxes, points_xy)

        if self._backend == "kit":
            self._set_kit_image(self._live_provider, live)
            self._set_kit_image(self._heatmap_provider, heat_view)
            self._set_kit_image(self._overlay_provider, overlay)
            for window in (self._live_window, self._heatmap_window, self._overlay_window):
                if window is not None:
                    window.visible = True
            return

        if self._backend == "opencv":
            import cv2

            cv2.imshow(TABLE_LIVE_TITLE, cv2.cvtColor(live, cv2.COLOR_RGB2BGR))
            cv2.imshow(TABLE_HEATMAP_TITLE, cv2.cvtColor(heat_view, cv2.COLOR_RGB2BGR))
            cv2.imshow(TABLE_OVERLAY_TITLE, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)


def _find_main_viewport(ui) -> object | None:
    for name in _MAIN_VIEWPORT_NAMES:
        window = ui.Workspace.get_window(name)
        if window is not None:
            return window
    try:
        for window in ui.Workspace.get_windows():
            title = getattr(window, "title", None) or getattr(window, "name", "") or ""
            if "viewport" in str(title).lower():
                return window
    except Exception:
        pass
    return None


def _dock_camera_panels(live_window, heatmap_window, overlay_window) -> None:
    """Pin camera panels into a column on the right of the main scene view."""
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

        for window in (live_window, heatmap_window, overlay_window):
            window.visible = True
        if scene_view is None:
            print(
                "[TeleAssist] main Viewport not found; leaving camera panels floating",
                flush=True,
            )
            return

        try:
            live_window.dock_in(scene_view, ui.DockPosition.RIGHT, 0.30)
            await app.next_update_async()
            heatmap_window.dock_in(live_window, ui.DockPosition.BOTTOM, 0.55)
            await app.next_update_async()
            overlay_window.dock_in(heatmap_window, ui.DockPosition.BOTTOM, 0.45)
            await app.next_update_async()
            for window in (live_window, heatmap_window, overlay_window):
                window.visible = True
            heatmap_window.focus()
            print(
                f"[TeleAssist] docked live/heatmap/scan panels to the right of "
                f"'{getattr(scene_view, 'title', scene_view)}'",
                flush=True,
            )
        except Exception as error:
            print(f"[TeleAssist] dock failed ({error}); panels should still be visible", flush=True)

    asyncio.ensure_future(pin())
