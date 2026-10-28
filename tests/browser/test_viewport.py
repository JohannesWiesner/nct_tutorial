"""Browser regressions for viewport recovery and preserving graph positions."""

import json
from threading import Thread
import unittest

from playwright.sync_api import sync_playwright
from werkzeug.serving import WSGIRequestHandler, make_server

from src.app import app


class QuietRequestHandler(WSGIRequestHandler):
    def log_request(self, code="-", size="-"):
        pass


class ViewportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        playwright = sync_playwright().start()
        cls.addClassCleanup(playwright.stop)
        cls.browser = playwright.chromium.launch()
        cls.addClassCleanup(cls.browser.close)
        server = make_server(
            "127.0.0.1", 0, app.server, threaded=True,
            request_handler=QuietRequestHandler,
        )
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()

        def stop_server():
            server.shutdown()
            thread.join(timeout=5)
            server.server_close()

        cls.addClassCleanup(stop_server)
        cls.url = f"http://127.0.0.1:{server.server_port}"

    def setUp(self):
        self.context = self.browser.new_context(viewport={"width": 1280, "height": 900})
        self.addCleanup(self.context.close)
        self.page = self.context.new_page()
        self.errors = []
        self.page.on("pageerror", lambda error: self.errors.append(str(error)))
        self.page.goto(self.url, wait_until="networkidle")
        self.page.wait_for_function("window.cy && window.cy.nodes().length === 9")
        self.assert_nodes_visible(all_nodes=True)

    def tearDown(self):
        self.assertEqual(self.errors, [])

    def assert_nodes_visible(self, all_nodes=False):
        self.page.wait_for_function(
            """all => {
                const cy = window.cy;
                const inside = node => {
                    const point = node.renderedPosition();
                    return point.x >= 0 && point.x <= cy.width()
                        && point.y >= 0 && point.y <= cy.height();
                };
                const intersects = node => {
                    const box = node.renderedBoundingBox();
                    return box.x2 >= 0 && box.x1 <= cy.width()
                        && box.y2 >= 0 && box.y1 <= cy.height();
                };
                return cy.nodes().length > 0 && (all
                    ? cy.nodes().every(inside) : cy.nodes().some(intersects));
            }""",
            arg=all_nodes,
            timeout=3000,
        )

    def upload(self, adjacency):
        self.page.locator("#upload-data input").set_input_files(
            {"name": "network.json", "mimeType": "application/json",
             "buffer": json.dumps({"A": adjacency}).encode()}
        )
        self.page.wait_for_function(
            "count => window.cy.nodes().length === count", arg=len(adjacency)
        )
        # Cytoscape debounces its automatic layout refresh after add/remove.
        self.page.wait_for_timeout(250)

    def test_large_pan_recovers_graph(self):
        for x, y in ((-100000, 0), (100000, 0), (0, -100000), (0, 100000)):
            self.page.evaluate("pan => window.cy.pan(pan)", {"x": x, "y": y})
            self.assert_nodes_visible(all_nodes=True)

    def test_visible_pan_and_interaction_remain_enabled(self):
        expected = self.page.evaluate(
            """() => {
                const cy = window.cy;
                cy.panBy({x: 10, y: 10});
                return cy.pan();
            }"""
        )
        self.page.wait_for_timeout(150)
        self.assertEqual(self.page.evaluate("window.cy.pan()"), expected)
        self.assertTrue(self.page.evaluate("window.cy.userPanningEnabled()"))
        self.assertTrue(self.page.evaluate("window.cy.userZoomingEnabled()"))

    def test_extreme_zoom_is_bounded(self):
        bounds = self.page.locator("#cytoscape").bounding_box()
        self.page.mouse.move(bounds["x"] + bounds["width"] / 2, bounds["y"] + 200)
        for delta in (100000, -100000):
            self.page.mouse.wheel(0, delta)
            self.page.wait_for_timeout(200)
            self.assert_nodes_visible()
            minimum, zoom, maximum = self.page.evaluate(
                "[window.cy.minZoom(), window.cy.zoom(), window.cy.maxZoom()]"
            )
            self.assertGreater(minimum, 1e-12)
            self.assertGreaterEqual(zoom, minimum)
            self.assertLessEqual(zoom, maximum)
            self.assertLessEqual(maximum, 4)

    def test_empty_space_between_disconnected_nodes_recovers(self):
        self.upload([[0, 0], [0, 0]])
        self.page.evaluate(
            """() => {
                const cy = window.cy;
                cy.nodes()[0].position({x: -1000, y: 0});
                cy.nodes()[1].position({x: 1000, y: 0});
                cy.viewport({zoom: 1, pan: {x: cy.width() / 2, y: cy.height() / 2}});
            }"""
        )
        self.assert_nodes_visible(all_nodes=True)

    def test_single_node_drag_outside_recovers(self):
        self.upload([[0]])
        self.page.evaluate(
            """() => {
                const node = window.cy.nodes()[0];
                node.position({x: 100000, y: -100000});
                node.emit('dragfree');
            }"""
        )
        self.assert_nodes_visible(all_nodes=True)

    def test_resize_and_fit_show_all_nodes(self):
        for width in (360, 1800, 480):
            self.page.set_viewport_size({"width": width, "height": 900})
            self.assert_nodes_visible()
            self.page.locator("#fit-graph").click()
            self.assert_nodes_visible(all_nodes=True)

    def test_fit_preserves_node_positions_and_selection(self):
        before = self.page.evaluate(
            """() => {
                const cy = window.cy;
                const node = cy.nodes()[0];
                node.select();
                cy.zoom({level: 4, renderedPosition: node.renderedPosition()});
                return cy.nodes().map(n => ({id: n.id(), position: n.position()}));
            }"""
        )
        self.page.locator("#fit-graph").click()
        self.assert_nodes_visible(all_nodes=True)
        after = self.page.evaluate(
            "window.cy.nodes().map(n => ({id: n.id(), position: n.position()}))"
        )
        self.assertEqual(after, before)
        self.assertEqual(self.page.evaluate("window.cy.nodes(':selected').length"), 1)

    def test_state_swap_preserves_dragged_positions_selection_and_layout(self):
        before = self.page.evaluate(
            """() => {
                const cy = window.cy;
                const node = cy.nodes()[0];
                node.position({x: node.position('x') + 10, y: node.position('y') + 10});
                node.select();
                window.stateSwapLayoutRuns = 0;
                cy.on('layoutstart', () => window.stateSwapLayoutRuns++);
                return cy.nodes().map(n => ({id: n.id(), position: n.position()}));
            }"""
        )
        self.page.locator("#switch-states").click()
        self.page.wait_for_function(
            "document.querySelector('#edit-status').textContent.includes('Switched x0 and xf')"
        )
        # A layout would be triggered after the render callback and its debounce.
        self.page.wait_for_timeout(500)
        after = self.page.evaluate(
            "window.cy.nodes().map(n => ({id: n.id(), position: n.position()}))"
        )
        self.assertEqual(after, before)
        self.assertEqual(self.page.evaluate("window.cy.nodes(':selected').length"), 1)
        self.assertEqual(self.page.evaluate("window.stateSwapLayoutRuns"), 0)


if __name__ == "__main__":
    unittest.main()
