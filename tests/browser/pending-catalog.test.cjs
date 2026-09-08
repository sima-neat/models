const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const {test} = require('node:test');
const source = fs.readFileSync(`${__dirname}/../../artifact-browser/app.js`, 'utf8');
function extract(name) {
  const start = source.indexOf(`function ${name}(`);
  return source.slice(start, source.indexOf('\n}\n', start) + 2);
}
test('pending model remains visible without offering a nonexistent download', () => {
  const model = {model_id: 'new_model', display_name: 'New model', builds: [], pending_builds: [{}]};
  const context = vm.createContext({
    state: {models: [model]}, filters: () => ({query: ''}), normalizedSearchTerms: () => [],
    modelSearchDocument: () => '', profileGroups: () => [], activeModelBuild: () => null, escapeHtml: value => value,
  });
  vm.runInContext(extract('filteredModels') + '\n' + extract('renderModelRow'), context);
  assert.equal(context.filteredModels().length, 1);
  const html = context.renderModelRow(model);
  assert.match(html, /Awaiting compiled builds/);
  assert.match(html, /disabled/);
  assert.ok(!html.includes('data-run-id'));
});
test('pending-only models do not pretend to match a compiler filter', () => {
  const context = vm.createContext({
    state: {models: [{builds: [], pending_builds: [{}]}]}, filters: () => ({query: '', compiler: '2.1.3'}),
    normalizedSearchTerms: () => [], modelSearchDocument: () => '',
  });
  vm.runInContext(extract('filteredModels'), context);
  assert.equal(context.filteredModels().length, 0);
});
