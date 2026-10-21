// F5 syntax check: reads a JSON array of source texts on stdin, prints a JSON array of parse-diagnostic counts.
// "inner" = errors located before the end of the text; "eof" = errors at the very end (caused by truncation, e.g.
// unclosed blocks). Uses the TypeScript parser only (no type checking). TypeScript module path: argv[2].
const ts = require(process.argv[2]);
let input = "";
process.stdin.on("data", (c) => (input += c));
process.stdin.on("end", () => {
  const out = JSON.parse(input).map((text) => {
    const sf = ts.createSourceFile("a.ts", text, ts.ScriptTarget.Latest, false, ts.ScriptKind.TS);
    const diags = sf.parseDiagnostics || [];
    const inner = diags.filter((d) => d.start < text.length).length;
    return { inner, eof: diags.length - inner };
  });
  process.stdout.write(JSON.stringify(out));
});
