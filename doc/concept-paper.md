# Concept

## 3d Context

### Program
- Grid and a Front z plane, and at the Back in Z direction a sink-pyramid --- like functioncall but inner view
- Base and Sink dynamically expand to match/fit the inner ast. In order to fit wide asts "fill" the sink part so the ast area is a trapezoid.
- only exactly 1 node may be connected to the sink

### Sink
- sink preexists
- only 1
- exactly 1 node must be connected to the sink -> otherwise error
- just infers the type

### Match
- matchfront (plane) and 1 or several asymmetric pyramids as match sinks at the end.
- expands over several "levels" wir providing grids for forther sub expressions for each type match case
- a match can have mor than 1 output -> several match sinks side by side
- each match sink must be connected to exactly 1 node at each inner level -> otherwise error
- a match sink infers its type as the sumtype of all inputs
- each match grid gets 1 pattern node at the front as input
- expressions of the match grids can also accept edges from outside grids/scopes
- all patterns of a match must must cover the whole input type.

### VarDecl
- cube-node at the front plane
- only typeclass as type, no const type
- lateral order matters = arg index for call inside the ide or later for cli args

### ConstDecl
- cube-node not at the front plane
- must have concrete const type as type

### FuncCall
- lying pyramide
- front expands with number of function parameters/anchors.
- input anchors are ordered by parameter index from left to right.

### TypeCast
- cube node with input and output anchor
- can have typeclass or const type
- if typecast can fail -> output type is sumtype with undefined

### Types?????
- Color coding of the edges
- sum types: multiple edges stacked on on another in y-dimension for each sub type ->
- overflow strategies: collapse/fadeout?????????
- PROBLEM WITH COLORS: black/white printing? color blind? not intuitive?
- better: types using x and y dimensions
- const type?
- problem of cluttering if too complex
- dotted/dashed lines?
- ===> umschalten zwischen anzeigearten??!!!!!!!!
- vergleichen argumentieren
- oder sonst "future work"
- meta studien finden?
- eventuell hilfe von Christoph



# layouting
wenn dynamisch nicht geht dann egal für arbeit
maximal lästig aber nicht zwingend nötig




# science os
