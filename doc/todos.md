# TODOs

- a run narrows every anchor it reaches to the literal it produced, but the
  arms it went past stay drawn in full. A branch the Match did not select is
  as settled as the row it did — fade away/remove/collapse?
- a source literal is classified by its base type, so `1 -> Bool` is partial
  although it always succeeds. Refining it would mean evaluating at inference
  time; if that is ever wanted, it belongs beside `cast_kind` and nowhere else
