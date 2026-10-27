const assert = require("node:assert/strict");
const fs = require("node:fs");
const vm = require("node:vm");
const test = require("node:test");
const path = require("node:path");

function app() {
  const nodes = new Map();
  const document = {
    documentElement: { dataset: {} },
    querySelector(selector) {
      if (!nodes.has(selector)) nodes.set(selector, {
        dataset: {}, style: {}, innerHTML: "", checked: true,
        classList: { add() {}, remove() {}, toggle() {} },
        addEventListener() {}, setAttribute() {}, removeAttribute() {},
        scrollIntoView() {}, querySelectorAll() { return []; },
      });
      return nodes.get(selector);
    },
    querySelectorAll() { return []; },
  };
  const context = vm.createContext({
    document, FormData, URL, localStorage: { getItem() { return null; } },
    matchMedia() { return { matches: false }; },
  });
  const source = fs.readFileSync(path.join(__dirname,
    "../src/deepfake_detection/webapp/static/app.js"), "utf8");
  vm.runInContext(source.slice(0, source.lastIndexOf("loadConfig().then(")), context);
  vm.runInContext(`state.config = { samples: [
    { id: "real", title: "Real", truth: "real", dataset: "fixture" },
    { id: "fake", title: "Fake", truth: "fake", dataset: "fixture" }
  ], operating_points: { ensemble: { thresholds: [0.5], datasets: {} } } };
  state.model = "ensemble";`, context);
  return { context, nodes, run: (code) => vm.runInContext(code, context) };
}

test("switching samples discards the pending result", async () => {
  const { context, run } = app();
  let finish;
  context.fetch = async (_url, options) => {
    assert.equal(options.body.get("sample"), "real");
    return new Promise((resolve) => { finish = resolve; });
  };
  run("renderResult = () => {}; chooseSample('real');");
  const pending = run("analyze()");
  run("chooseSample('fake')");
  finish({ ok: true, json: async () => ({ probability: 0.1 }) });
  await pending;
  assert.equal(run("state.analysis"), null);
});

test("an earlier response cannot overwrite a newer request", async () => {
  const { context, run } = app();
  const responses = [];
  context.fetch = () => new Promise((resolve) => responses.push(resolve));
  run("renderResult = () => {}; chooseSample('real');");
  const first = run("analyze()");
  run("chooseSample('fake')");
  const second = run("analyze()");
  responses[1]({ ok: true, json: async () => ({ probability: 0.9 }) });
  await second;
  responses[0]({ ok: true, json: async () => ({ probability: 0.1 }) });
  await first;
  assert.equal(run("state.analysis.probability"), 0.9);
  assert.equal(run("state.analysis.sample.truth"), "fake");
});

test("empty operating-point datasets show unavailable rates", () => {
  const { nodes, run } = app();
  run(`state.analysis = { probability: 0.8, model: "ensemble", members: {}, blockers: [] };
    refreshThreshold();`);
  assert.match(nodes.get("#rates").innerHTML, /No per-threshold record/);
});
