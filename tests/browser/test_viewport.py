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
        self.wait_arranged()
        self.assert_nodes_visible(all_nodes=True)

    def tearDown(self):
        self.assertEqual(self.errors, [])

    def wait_arranged(self):
        self.page.wait_for_function(
            """() => window.dash_clientside.nctArrangement
                && !window.dash_clientside.nctArrangement.busy(window.cy)
                && document.querySelector('#layout-status').textContent === 'Graph arranged.'"""
        )

    def request_arrangement(self):
        self.page.locator("#arrange-graph").click()
        self.page.wait_for_function("window.dash_clientside.nctArrangement.busy(window.cy)")

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
        self.wait_arranged()

    def test_arrange_recovers_messy_graph_preserving_data_and_selection(self):
        before = self.page.evaluate(
            """() => {
                const cy = window.cy;
                cy.nodes()[0].select();
                cy.edges()[0].select();
                cy.nodes().forEach((n, i) => n.position({x: i * 100000, y: 0}));
                return {data: cy.elements().map(e => e.data()),
                    selected: cy.$(':selected').map(e => e.id())};
            }"""
        )
        self.request_arrangement()
        self.wait_arranged()
        after = self.page.evaluate(
            """() => ({data: window.cy.elements().map(e => e.data()),
                selected: window.cy.$(':selected').map(e => e.id())})"""
        )
        self.assertEqual(after, before)
        self.assertLess(self.page.evaluate("window.cy.nodes().boundingBox().w"), 5000)
        self.assertTrue(self.page.evaluate(
            """() => {
                const boxes = window.cy.nodes().map(n => n.boundingBox());
                return boxes.every((a, i) => boxes.slice(i + 1).every(b =>
                    a.x2 <= b.x1 || a.x1 >= b.x2 || a.y2 <= b.y1 || a.y1 >= b.y2));
            }"""
        ))
        self.assert_nodes_visible(all_nodes=True)

    def test_wheel_cancels_animation_without_late_fit(self):
        bounds = self.page.locator("#cytoscape").bounding_box()
        self.request_arrangement()
        self.page.mouse.move(bounds["x"] + bounds["width"] / 2, bounds["y"] + 200)
        self.page.mouse.wheel(0, -50)
        self.page.wait_for_function(
            "document.querySelector('#layout-status').textContent === 'Arrangement stopped.'"
        )
        self.page.wait_for_timeout(100)
        snapshot = """() => ({positions: window.cy.nodes().map(n => n.position()),
            pan: window.cy.pan(), zoom: window.cy.zoom()})"""
        before = self.page.evaluate(snapshot)
        self.page.wait_for_timeout(500)
        self.assertEqual(self.page.evaluate(snapshot), before)

    def test_repeated_arrangement_upload_and_resize_leave_only_current_graph(self):
        self.request_arrangement()
        self.request_arrangement()
        self.upload([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
        for width in (320, 1800, 480):
            self.request_arrangement()
            self.page.set_viewport_size({"width": width, "height": 900})
            self.wait_arranged()
            self.assertEqual(self.page.evaluate("window.cy.nodes().map(n => n.id())"), ["0", "1", "2"])
            self.assertEqual(self.page.evaluate("window.cy.edges().length"), 2)
            self.assert_nodes_visible(all_nodes=True)

    def test_weight_edit_preserves_finished_arrangement(self):
        self.request_arrangement()
        self.wait_arranged()
        before = self.page.evaluate(
            """() => {
                window.cy.edges()[0].select();
                return window.cy.nodes().map(n => n.position());
            }"""
        )
        self.page.wait_for_timeout(200)
        self.page.locator("#edge-weight").fill("4.5")
        self.page.locator("#set-edge-weight").click()
        self.page.wait_for_function("window.cy.edges()[0].data('weight') === 4.5")
        self.page.wait_for_timeout(500)
        self.assertEqual(self.page.evaluate("window.cy.nodes().map(n => n.position())"), before)

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

    def test_role_toggles_preserve_positions_selection_viewport_and_layout(self):
        self.request_arrangement()
        self.wait_arranged()
        before = self.page.evaluate(
            """() => {
                const cy = window.cy;
                const node = cy.getElementById('0');
                node.position({x: node.position('x') + 10, y: node.position('y') + 10});
                node.emit('dragfree');
                node.select();
                cy.getElementById('2').select();
                window.roleLayoutRuns = 0;
                cy.on('layoutstart', () => window.roleLayoutRuns++);
                window.roleSnapshot = () => ({
                    positions: cy.nodes().map(n => ({id: n.id(), position: n.position()})),
                    selected: cy.nodes(':selected').map(n => n.id()),
                    pan: cy.pan(), zoom: cy.zoom(), count: cy.elements().length
                });
                return window.roleSnapshot();
            }"""
        )
        # Allow Cytoscape's drag/selection props to reach Dash before clicking.
        self.page.wait_for_timeout(200)
        for button, expected in (
            ("toggle-controller", [0, 1]), ("toggle-constraint", [0, 0]),
            ("toggle-controller", [1, 0]), ("toggle-constraint", [1, 1]),
        ):
            self.page.locator(f"#{button}").click()
            self.page.wait_for_function(
                """expected => ['0', '2'].every(id => {
                    const node = window.cy.getElementById(id);
                    const opacity = node.pstyle('background-image-opacity').value;
                    return opacity[0] === expected[0] && opacity[1] === expected[1];
                })""", arg=expected,
            )
            self.page.wait_for_timeout(200)
            self.assertEqual(self.page.evaluate("window.roleSnapshot()"), before)
            self.assertEqual(self.page.evaluate("window.roleLayoutRuns"), 0)
            self.assertEqual(self.page.evaluate(
                "window.cy.getElementById('1').pstyle('background-image-opacity').value"
            ), [1, 1])

    def test_fit_includes_badges_and_selection_outline(self):
        self.page.evaluate("window.cy.nodes().select()")
        for width in (320, 768, 1280, 1920):
            self.page.set_viewport_size({"width": width, "height": 900})
            self.page.locator("#fit-graph").click()
            self.page.wait_for_function(
                """() => {
                    const cy = window.cy;
                    return cy.nodes().every(node => {
                        const box = node.renderedBoundingBox();
                        return box.x1 >= 0 && box.y1 >= 0
                            && box.x2 <= cy.width() && box.y2 <= cy.height();
                    });
                }"""
            )

    def test_right_hand_controls_fit_without_horizontal_scrolling(self):
        for width in (320, 375, 768, 992, 1024, 1280, 1920):
            with self.subTest(width=width):
                self.page.set_viewport_size({"width": width, "height": 900})
                self.page.wait_for_timeout(200)
                failures = self.page.evaluate(
                    """() => {
                        const selectors = [
                            '#network-controls', '#network-controls section',
                            '#states-table .dash-spreadsheet-container', '#states-table table',
                            '.parameter-grid', '.parameter-field', '.network-toolbar',
                            '.edge-weight-control', '#network-controls button',
                            '#network-controls input', '#network-controls select'
                        ];
                        return [...document.querySelectorAll(selectors.join(','))].flatMap(el => {
                            const rect = el.getBoundingClientRect();
                            const outside = rect.left < -1 || rect.right > innerWidth + 1;
                            const scrolls = el.clientWidth > 0 && el.scrollWidth > el.clientWidth + 1;
                            return outside || scrolls ? [{
                                element: el.id || el.className,
                                left: rect.left, right: rect.right,
                                clientWidth: el.clientWidth, scrollWidth: el.scrollWidth
                            }] : [];
                        });
                    }"""
                )
                self.assertEqual(failures, [])
        # Narrow screens still allow editing both state columns and parameters.
        self.page.set_viewport_size({"width": 320, "height": 900})
        for column, value in (("x0", "2.25"), ("xf", "3.5")):
            cell = self.page.locator(
                f'#states-table td[data-dash-row="0"][data-dash-column="{column}"]'
            )
            cell.dblclick()
            cell.locator("input").fill(value)
            cell.locator("input").press("Enter")
            self.assertIn(value, cell.inner_text())
        self.page.locator("#T").fill("1.25")
        self.page.locator("#switch-states").click()
        self.page.wait_for_function(
            "document.querySelector('#edit-status').textContent.includes('Switched x0 and xf')"
        )
        self.assertEqual(self.page.locator("#T").input_value(), "1.25")


if __name__ == "__main__":
    unittest.main()
