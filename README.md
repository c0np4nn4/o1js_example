# o1js example

## how to run

- env: Node.js (`v20.11.1`)

```bash
npm run build #translate `index.ts` to `index.js`
npm run start
```

- you can compare the result from the code using `sklearn`.

```bash
python3 ./src/sample_data/sklearn_test.py
```

## Result

```bash
# o1js
> o1js_example@1.0.0 start
> node dist/index.js

start
making proof
proof created
value:  525
Proof is valid: true

# ------------------------------------

# python3 (sklearn)
Input: 25, Predicted Target: 544.5179222010554

```
