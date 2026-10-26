const assert = require("node:assert/strict");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const cytoscape = require("cytoscape");

const source = readFileSync(join(__dirname, "../assets/viewport.js"), "utf8");

function fixture(t, positions = [{x: 0, y: 0}, {x: 100, y: 100}]) {
    const cy = cytoscape({
        headless: true,
        styleEnabled: true,
        maxZoom: 4,
        layout: {name: "preset"},
        elements: positions.map((position, i) => ({data: {id: String(i)}, position}))
    });
    // Supply only the DOM geometry missing in headless mode; viewport and
    // bounding-box calculations use the real Cytoscape.js implementation.
    const size = {width: 800, height: 520};
    cy.width = () => size.width;
    cy.height = () => size.height;
    const originalContainer = cy.container;
    cy.container = () => ({id: "cytoscape"});
    t.after(() => { cy.container = originalContainer; cy.destroy(); });
    const window = {
        cy,
        dash_clientside: {no_update: Symbol("no update"), callback_context: {triggered: []}}
    };
    vm.runInNewContext(source, {window});
    const guard = (clicks = 0) => window.dash_clientside.nctViewport.keepVisible(null, null, clicks);
    guard();
    return {cy, size, window, guard};
}

function allNodesInside(cy) {
    return cy.nodes().every(node => {
        const {x, y} = node.renderedPosition();
        return x >= 0 && x <= cy.width() && y >= 0 && y <= cy.height();
    });
}

test("large pans in every direction recover the graph", t => {
    const {cy, guard} = fixture(t);
    for (const [x, y] of [[-100000, 0], [100000, 0], [0, -100000], [0, 100000]]) {
        cy.pan({x, y});
        assert.equal(allNodesInside(cy), false);
        guard();
        assert.ok(allNodesInside(cy));
    }
});

test("normal pan and zoom remain available without snapping", t => {
    const {cy, guard} = fixture(t);
    cy.panBy({x: 10, y: 10});
    const pan = {...cy.pan()};
    guard();
    assert.deepEqual(cy.pan(), pan);
    assert.ok(cy.userPanningEnabled());
    assert.ok(cy.userZoomingEnabled());
});

test("zoom is bounded while large graphs can still fit", t => {
    const {cy, guard} = fixture(t);
    cy.zoom(1e-20);
    guard();
    assert.ok(cy.zoom() >= cy.minZoom());
    cy.zoom(1e20);
    guard();
    assert.ok(cy.zoom() <= 4);
    cy.nodes()[1].position({x: 100000, y: 100000});
    guard();
    assert.ok(cy.minZoom() < 0.01);
    cy.fit(cy.nodes(), 32);
    assert.ok(allNodesInside(cy));
});

test("empty space inside the overall bounding box is recovered", t => {
    const {cy, guard} = fixture(t, [{x: -1000, y: 0}, {x: 1000, y: 0}]);
    cy.viewport({zoom: 1, pan: {x: cy.width() / 2, y: cy.height() / 2}});
    assert.equal(cy.nodes().some(node => {
        const {x} = node.renderedPosition();
        return x >= 0 && x <= cy.width();
    }), false);
    guard();
    assert.ok(allNodesInside(cy));
});

test("dragging a lone node outside the canvas recovers it", t => {
    const {cy, guard} = fixture(t, [{x: 0, y: 0}]);
    cy.nodes()[0].position({x: 100000, y: -100000});
    assert.equal(allNodesInside(cy), false);
    guard();
    assert.ok(allNodesInside(cy));
});

test("Fit graph preserves node positions and selection", t => {
    const {cy, window, guard} = fixture(t);
    cy.nodes()[0].select();
    const positions = cy.nodes().map(node => ({...node.position()}));
    cy.zoom({level: 4, renderedPosition: cy.nodes()[0].renderedPosition()});
    window.dash_clientside.callback_context.triggered = [{prop_id: "fit-graph.n_clicks"}];
    guard(1);
    assert.ok(allNodesInside(cy));
    assert.deepEqual(cy.nodes().map(node => ({...node.position()})), positions);
    assert.equal(cy.nodes(":selected").length, 1);
});

test("a smaller canvas recovers the graph and then stays stable", t => {
    const {cy, size, guard} = fixture(t);
    size.width = 180;
    size.height = 180;
    cy.pan({x: 500, y: 500});
    guard();
    assert.ok(allNodesInside(cy));
    let events = 0;
    cy.on("viewport", () => { events += 1; });
    guard();
    guard();
    assert.equal(events, 0);
});

test("missing, empty, and hidden graphs defer recovery safely", t => {
    const {cy, size, window, guard} = fixture(t);
    window.cy = null;
    assert.doesNotThrow(() => guard());
    window.cy = cy;
    size.width = 0;
    assert.doesNotThrow(() => guard());
    size.width = 800;
    cy.elements().remove();
    assert.doesNotThrow(() => guard());
});
