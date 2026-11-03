const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const test = require("node:test");
const vm = require("node:vm");
const cytoscape = require("cytoscape");
const CytoscapeComponent = require("react-cytoscapejs").default;

// Exercise the actual Python stylesheet and Dash patch, not a copied fixture.
const output = spawnSync(process.env.PYTHON || "python", ["-c", `
import copy, json
import numpy as np
from src.app import render_network
from src.graph_utils import default_store
from src.view_utils import stylesheet_for
store = default_store(np.array([[0, 1, 4, 0], [1, 0, 0, 2], [4, 0, 0, 3], [0, 2, 3, 0]]))
for node, (controller, constrain) in zip(store["elements"]["nodes"],
                                       [(True, True), (True, False), (False, True), (False, False)]):
    node["data"].update(controller=controller, constrain=constrain)
    node["data"]["label"] = str(int(node["data"]["id"]) + 100)
current = list(reversed(store["elements"]["edges"] + store["elements"]["nodes"]))
modified = copy.deepcopy(store)
for node in modified["elements"]["nodes"]:
    node["data"]["controller"] = not node["data"]["controller"]
    node["data"]["constrain"] = not node["data"]["constrain"]
patch = render_network(modified, current)[0].to_plotly_json()
print(json.dumps({"elements": current, "style": stylesheet_for(store["elements"]), "patch": patch}))
`], {cwd: join(__dirname, ".."), encoding: "utf8"});
assert.equal(output.status, 0, output.stderr);
const fixture = JSON.parse(output.stdout);

function graph(t) {
    const cy = cytoscape({
        headless: true, styleEnabled: true, layout: {name: "preset"},
        elements: structuredClone(fixture.elements), style: fixture.style,
    });
    const originalContainer = cy.container;
    t.after(() => { cy.container = originalContainer; cy.destroy(); });
    return cy;
}

test("all four roles use distinct badge combinations with identical node geometry", t => {
    const cy = graph(t);
    const expected = [[1, 1], [1, 0], [0, 1], [0, 0]];
    const bounds = [];
    for (let i = 0; i < 4; i++) {
        const node = cy.getElementById(String(i));
        assert.deepEqual(node.pstyle("background-image-opacity").value, expected[i]);
        assert.equal(node.width(), 48);
        assert.equal(node.height(), 48);
        assert.equal(node.style("label"), String(i + 100));
        assert.equal(node.style("text-valign"), "center");
        assert.equal(node.style("text-halign"), "center");
        assert.equal(node.style("border-style"), "solid");
        const box = node.boundingBox();
        // Badges reach 7 px above and 5 px beside the 48 px body.
        assert.ok(box.y1 <= node.position("y") - 31);
        assert.ok(box.x1 <= node.position("x") - 29);
        assert.ok(box.x2 >= node.position("x") + 29);
        bounds.push([box.w, box.h]);
    }
    assert.ok(bounds.every(box => JSON.stringify(box) === JSON.stringify(bounds[0])));
    assert.equal(cy.nodes().length, 4);
    assert.equal(cy.edges().length, 4);
});

test("badge icons retain their size, fixed corners, and unclipped opaque backgrounds", t => {
    const cy = graph(t);
    const node = cy.nodes()[0];
    const urls = node.pstyle("background-image").value;
    assert.equal(urls.length, 2);
    for (const url of urls) {
        const svg = decodeURIComponent(url.slice(url.indexOf(",") + 1));
        assert.match(svg, /<svg[^>]*width="20" height="20"/);
        assert.match(svg, /<circle[^>]*fill="white"/);
    }
    assert.notEqual(urls[0], urls[1]);
    assert.deepEqual(node.pstyle("background-position-x").value, [0, 100]);
    assert.deepEqual(node.pstyle("background-offset-x").value, [-5, 5]);
    assert.deepEqual(node.pstyle("background-offset-y").value, [-7, -7]);
    assert.deepEqual(node.pstyle("background-clip").value, ["none", "none"]);
    assert.deepEqual(node.pstyle("background-image-containment").value, ["over", "over"]);
});

test("selection adds an independent outline without changing role badges or body size", t => {
    const cy = graph(t);
    for (const node of cy.nodes()) {
        const opacity = node.pstyle("background-image-opacity").value;
        const border = node.style("border-color");
        node.select();
        assert.deepEqual(node.pstyle("background-image-opacity").value, opacity);
        assert.equal(node.width(), 48);
        assert.notEqual(node.style("border-color"), border);
        assert.ok(node.numericStyle("underlay-opacity") > 0);
    }
});

test("Fit graph includes badge extents at narrow and wide canvas sizes", t => {
    const cy = graph(t);
    cy.container = () => ({id: "cytoscape"});
    const window = {cy, dash_clientside: {
        no_update: Symbol("no update"),
        callback_context: {triggered: [{prop_id: "fit-graph.n_clicks"}]},
    }};
    vm.runInNewContext(readFileSync(join(__dirname, "../assets/viewport.js"), "utf8"), {window});
    cy.nodes().select();
    for (const width of [280, 800, 1600]) {
        cy.width = () => width;
        cy.height = () => 520;
        window.dash_clientside.nctViewport.keepVisible(null, null, 1);
        for (const node of cy.nodes()) {
            const box = node.renderedBoundingBox();
            assert.ok(box.x1 >= 31 && box.y1 >= 31);
            assert.ok(box.x2 <= width - 31 && box.y2 <= 489);
        }
    }
});

test("edge labels are absent while weight data and scaled widths remain", t => {
    const cy = graph(t);
    assert.ok(cy.edges().every(edge => edge.style("label") === ""));
    assert.equal(cy.getElementById("0--1").data("weight"), 1);
    assert.equal(cy.getElementById("0--2").data("weight"), 4);
    assert.ok(cy.getElementById("0--2").numericStyle("width") > cy.getElementById("0--1").numericStyle("width"));
});

test("React applies the actual role patch without moving nodes, changing selection, or running a layout", t => {
    const cy = graph(t);
    const component = new CytoscapeComponent({});
    component._cy = cy;
    const previous = {
        ...CytoscapeComponent.defaultProps,
        elements: structuredClone(fixture.elements), stylesheet: fixture.style,
        layout: {name: "preset"},
    };
    const elements = structuredClone(previous.elements);
    for (const {location, operation, params} of fixture.patch.operations) {
        assert.equal(operation, "Assign");
        let target = elements;
        for (const key of location.slice(0, -1)) target = target[key];
        target[location.at(-1)] = params.value;
    }
    // Positions/selection change locally after the server reads the props.
    cy.nodes()[0].position({x: 80, y: 160}).select();
    cy.zoom(1.2);
    cy.pan({x: 20, y: 15});
    const snapshot = () => ({
        positions: cy.nodes().map(n => ({id: n.id(), position: {...n.position()}})),
        selected: cy.nodes(":selected").map(n => n.id()),
        pan: {...cy.pan()}, zoom: cy.zoom(), count: cy.elements().length,
    });
    const before = snapshot();
    let layoutRuns = 0;
    cy.on("layoutstart", () => layoutRuns++);
    component.updateCytoscape(previous, {...previous, elements});
    assert.deepEqual(snapshot(), before);
    assert.equal(layoutRuns, 0);
    for (const element of elements.filter(e => e.data.source === undefined)) {
        const node = cy.getElementById(element.data.id);
        assert.equal(node.data("controller"), element.data.controller);
        assert.equal(node.data("constrain"), element.data.constrain);
        assert.deepEqual(node.pstyle("background-image-opacity").value,
            [Number(element.data.controller), Number(element.data.constrain)]);
    }
});
