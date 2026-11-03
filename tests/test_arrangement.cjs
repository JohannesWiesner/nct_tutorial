const assert = require("node:assert/strict");
const test = require("node:test");
const {graph, fixture, network, positions, overlaps, assertVisible, until, pause} = require("./graph_test_helpers.cjs");

test("initial arrangement fits badge bounds and saves finished positions", async t => {
    const {cy, guard, settled, events} = graph(t);
    guard();
    await settled();
    assert.equal(events.runs, 1);
    assert.equal(events.fits, 1);
    assert.equal(events.status, "Graph arranged.");
    assert.equal(overlaps(cy), 0);
    assertVisible(cy);
    const saved = events.writes.findLast(item => item.id === "cytoscape").props.elements;
    assert.deepEqual(saved.filter(e => e.position).map(e => ({id: e.data.id, ...e.position})), positions(cy));
});

test("Arrange recovers extreme/coincident positions without changing data, selection, or S roles", async t => {
    const {cy, guard, settled} = graph(t);
    guard(); await settled();
    cy.nodes()[0].select(); cy.edges()[0].select();
    const selected = cy.$(":selected").map(e => e.id());
    const data = cy.elements().map(e => structuredClone(e.data()));
    for (const coincident of [false, true]) {
        cy.nodes().forEach((node, i) => node.position({x: coincident ? 0 : i * 1e6, y: 0}));
        guard("arrange-graph.n_clicks"); await settled();
        assert.equal(overlaps(cy), 0);
        assertVisible(cy);
        assert.ok(cy.nodes().boundingBox().w < 5000);
        assert.deepEqual(cy.elements().map(e => e.data()), data);
        assert.deepEqual(cy.$(":selected").map(e => e.id()), selected);
    }
});

test("Fit alone preserves positions and does not restart the layout", async t => {
    const {cy, guard, settled, events} = graph(t);
    guard(); await settled();
    cy.nodes()[0].position({x: 20000, y: 20000});
    const before = positions(cy), runs = events.runs;
    guard("fit-graph.n_clicks"); await pause(30);
    assert.deepEqual(positions(cy), before);
    assert.equal(events.runs, runs);
    assertVisible(cy);
});

for (const gesture of ["grab", "tapstart", "dragpan", "scrollzoom", "pinchzoom"]) {
    test(`${gesture} cancels animation with no late fit or coordinate changes`, async t => {
        const {cy, guard, settled, events, arrangement} = graph(t);
        guard(); await settled();
        guard("arrange-graph.n_clicks");
        await until(() => events.runs === 2);
        await pause(50);
        cy.emit(gesture);
        assert.equal(arrangement.busy(cy), false);
        const before = positions(cy), fits = events.fits;
        await pause(450);
        assert.deepEqual(positions(cy), before);
        assert.equal(events.fits, fits);
        assert.equal(events.status, "Arrangement stopped.");
    });
}

test("viewport protection is deferred during layout and restored after cancellation", async t => {
    const {cy, guard, settled, events, arrangement} = graph(t);
    guard(); await settled();
    guard("arrange-graph.n_clicks");
    const fits = events.fits;
    cy.pan({x: 1e6, y: 1e6}); guard();
    assert.equal(events.fits, fits);
    arrangement.cancel(cy); guard();
    assert.equal(events.fits, fits + 1);
    assertVisible(cy);
});

test("native fits and renderer resize/pan cannot interrupt an active or queued layout", async t => {
    const {cy, guard, settled, events, size, arrangement} = graph(t);
    guard();
    cy.fit(); cy.panBy({x: 10, y: 10}); cy.emit("resize");
    assert.equal(events.fits, 0);
    await settled();
    guard("arrange-graph.n_clicks");
    await until(() => events.runs === 2);
    size.width = 320;
    cy.emit("resize"); cy.fit(); guard();
    assert.equal(arrangement.busy(cy), true);
    await settled();
    assert.equal(events.fits, 2);
    assertVisible(cy);
});

test("repeated requests replace the old job and fit only the winner", async t => {
    const {cy, guard, settled, events} = graph(t);
    guard(); await settled();
    guard("arrange-graph.n_clicks"); await until(() => events.runs === 2);
    guard("arrange-graph.n_clicks"); guard("arrange-graph.n_clicks");
    await settled(); await pause(450);
    assert.equal(events.runs, 3);
    assert.equal(events.fits, 2);
    assertVisible(cy);
});

test("Fit and upload interrupt without delayed layout completion", async t => {
    const {cy, guard, settled, events} = graph(t);
    guard(); await settled();
    for (const trigger of ["fit-graph.n_clicks", "upload-data.contents"]) {
        const runs = events.runs;
        guard("arrange-graph.n_clicks"); await until(() => events.runs > runs);
        guard(trigger);
        const before = positions(cy), fits = events.fits;
        await pause(450);
        assert.deepEqual(positions(cy), before);
        assert.equal(events.fits, fits);
    }
});

test("batched structural edits start one replacement and never restore removed nodes", async t => {
    const {cy, guard, settled, events} = graph(t);
    guard(); await settled();
    guard("arrange-graph.n_clicks"); await until(() => events.runs === 2);
    cy.batch(() => {
        cy.nodes().slice(2).remove();
        cy.add({data: {id: "20", label: "20"}});
        cy.add({data: {id: "21", label: "21"}});
    });
    await settled(); await pause(450);
    assert.equal(events.runs, 3);
    assert.equal(events.fits, 2);
    assert.deepEqual(cy.nodes().map(n => n.id()), ["0", "1", "20", "21"]);
    assertVisible(cy);
});

test("a real server weight response cannot replay positions from before arrangement", async t => {
    const {cy, guard, settled, events, update} = graph(t);
    guard(); await settled();
    cy.nodes()[0].select();
    const before = positions(cy), runs = events.runs;
    update(fixture.weightUpdate);
    await pause(30);
    assert.deepEqual(positions(cy), before);
    assert.equal(events.runs, runs);
    assert.equal(cy.nodes()[0].selected(), true);
    assert.equal(cy.edges()[0].data("weight"), 4.5);
});

for (const [count, mode] of [[1, "isolated"], [12, "isolated"], [30, "disconnected"], [150, "connected"]]) {
    test(`${count}-node ${mode} graph has finite fitted positions and no badge overlap`, async t => {
        const elements = network(count, mode);
        const {cy, guard, settled} = graph(t, elements);
        const data = cy.elements().map(e => structuredClone(e.data()));
        guard(); await settled();
        assert.ok(positions(cy).every(p => Number.isFinite(p.x) && Number.isFinite(p.y)));
        assert.equal(overlaps(cy), 0);
        assert.deepEqual(cy.elements().map(e => e.data()), data);
        assertVisible(cy);
    });
}

test("empty and destroyed graphs do not retain pending work", async t => {
    const {cy, guard, settled, arrangement, events, destroy} = graph(t, []);
    guard(); await settled();
    assert.equal(events.runs, 0);
    arrangement.arrange(cy); destroy();
    await pause(20);
    assert.equal(arrangement.busy(cy), false);
    assert.equal(events.fits, 0);
});
