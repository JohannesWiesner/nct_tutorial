const assert = require("node:assert/strict");
const {spawnSync} = require("node:child_process");
const {readFileSync} = require("node:fs");
const {join} = require("node:path");
const vm = require("node:vm");
const cytoscape = require("cytoscape");
const CytoscapeComponent = require("react-cytoscapejs").default;
cytoscape.use(require("cytoscape-fcose"));

const output = spawnSync(process.env.PYTHON || "python", ["-c", `
import copy, json
from src.app import initial, render_network
from src.view_utils import stylesheet_for
edited = copy.deepcopy(initial)
edited["elements"]["edges"][0]["data"]["weight"] = 4.5
print(json.dumps({"elements": initial["elements"],
                  "style": stylesheet_for(initial["elements"]),
                  "weightUpdate": render_network(edited, initial["elements"])[0]}))
`], {cwd: join(__dirname, ".."), encoding: "utf8"});
assert.equal(output.status, 0, output.stderr);
const fixture = JSON.parse(output.stdout);
const flatten = elements => Array.isArray(elements) ? elements : [...elements.nodes, ...elements.edges];
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));
async function until(predicate, timeout = 6000) {
    const end = Date.now() + timeout;
    while (!predicate()) {
        assert.ok(Date.now() < end, "Timed out waiting for layout");
        await pause(10);
    }
}
function positions(cy) {
    return cy.nodes().map(node => ({id: node.id(), ...node.position()}));
}
function overlaps(cy) {
    const boxes = cy.nodes().map(node => node.boundingBox());
    let count = 0;
    boxes.forEach((a, i) => boxes.slice(i + 1).forEach(b => {
        if (a.x1 < b.x2 && a.x2 > b.x1 && a.y1 < b.y2 && a.y2 > b.y1) count++;
    }));
    return count;
}
function assertVisible(cy) {
    for (const node of cy.nodes()) {
        const b = node.renderedBoundingBox();
        assert.ok(b.x1 >= 0 && b.y1 >= 0 && b.x2 <= cy.width() && b.y2 <= cy.height(),
            `Node ${node.id()} is outside the viewport`);
    }
}
function network(count, mode = "connected") {
    const nodes = Array.from({length: count}, (_, i) => ({
        data: {id: String(i), label: String(i), controller: i % 2 === 0, constrain: i % 3 === 0},
        position: {x: i * 10000, y: (i % 3) * 10000},
    }));
    const edges = [];
    for (let i = 0; i < count; i++) {
        for (let j = i + 1; j < count; j++) {
            if (mode === "isolated" || (mode === "disconnected" && Math.floor(i / 5) !== Math.floor(j / 5))) continue;
            if (j === i + 1 || (i * 17 + j * 7) % 31 === 0) {
                edges.push({data: {id: `${i}--${j}`, source: String(i), target: String(j), weight: 1}});
            }
        }
    }
    return {nodes, edges};
}
function graph(t, elements = fixture.elements) {
    const cy = cytoscape({
        headless: true, styleEnabled: true, maxZoom: 4,
        layout: {name: "preset"}, elements: structuredClone(elements), style: fixture.style,
    });
    const size = {width: 800, height: 520};
    cy.width = () => size.width;
    cy.height = () => size.height;
    const originalContainer = cy.container;
    cy.container = () => ({id: "cytoscape"});
    const events = {runs: 0, fits: 0, writes: [], status: ""};
    cy.on("layoutstart", () => events.runs++);
    const originalFit = cy.fit;
    cy.fit = function (...args) { events.fits++; return originalFit.apply(cy, args); };
    const component = new CytoscapeComponent({});
    component._cy = cy;
    let previous = {
        ...CytoscapeComponent.defaultProps, elements: structuredClone(flatten(elements)),
        stylesheet: fixture.style, layout: {name: "preset", fit: false}, maxZoom: 4,
    };
    function update(elements) {
        const next = {...previous, elements: structuredClone(flatten(elements))};
        component.updateCytoscape(previous, next);
        previous = next;
    }
    const window = {cy, dash_clientside: {
        no_update: Symbol("no update"), callback_context: {triggered: []},
        set_props(id, props) {
            events.writes.push({id, props});
            if (id === "layout-status") events.status = props.children;
            if (id === "cytoscape") update(props.elements);
        },
    }};
    for (const file of ["viewport.js", "arrangement.js"]) {
        // Same realm as Cytoscape: its plain-object checks reject objects
        // originating in a separate vm context (unlike the real browser).
        vm.compileFunction(readFileSync(join(__dirname, "../assets", file), "utf8"),
            ["window", "setTimeout", "clearTimeout"])(window, setTimeout, clearTimeout);
    }
    const arrangement = window.dash_clientside.nctArrangement;
    function guard(trigger = "") {
        window.dash_clientside.callback_context.triggered = trigger ? [{prop_id: trigger}] : [];
        window.dash_clientside.nctViewport.keepVisible(null, null, 1, 1,
            trigger === "upload-data.contents" ? "data:application/json;base64,e30=" : null);
    }
    const settled = () => until(() => !arrangement.busy(cy));
    const destroy = () => { cy.container = originalContainer; cy.destroy(); };
    t.after(destroy);
    return {cy, size, window, events, arrangement, guard, settled, update, destroy};
}

module.exports = {graph, fixture, network, positions, overlaps, assertVisible, until, pause, cytoscape};
