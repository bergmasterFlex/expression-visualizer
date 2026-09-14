# TODOs

- evaluation step wise -> narrowing to literal types? -> also the match branches? fade away/remove/collapse?
- evaluation values rendered at correct positions for match/pattern/patternsink
- cast to a literal compares literal *text* in `infer::cast_kind` and parsed
  *values* in `eval::value_matches_type`, so a Constant typed `07` cast to `7`
  is `AlwaysNone` to the inferer and equal to the evaluator. Cure is
  canonicalising integer literals where they are typed, not a second rule here
- the thesis promises the type level of casting — distribution over sum types,
  refinement onto literal types — in `sec:cd:lang:inference` (see the `cbnote`
  in the appendix's casting section). It is implemented now and still unwritten
- a source literal is classified by its base type, so `1 -> Bool` is partial
  although it always succeeds. Refining it would mean evaluating at inference
  time; if that is ever wanted, it belongs beside `cast_kind` and nowhere else
