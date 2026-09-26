# TODOs

Stand 2026-09-22. Alles hier ist **Nachziehen an Kapitel 3 der Arbeit** — die Spezifikation
gilt, der Prototyp weicht ab. Die Arbeit geht durchgehend davon aus, dass es so implementiert
ist, wie Kapitel 3 es beschreibt; jede Abweichung unten ist also ein Fehler im Prototyp und
keine offene Entwurfsfrage.

Gefunden beim vollständigen Gegenlesen von Code gegen `03-concept-design.tex`
(siehe `../bsc-thesis/thesis-ch4-plan.md` §5 für die jeweilige Fundstelle in der Arbeit).

---

## 1. [!] Das Volumen ist transient, wo es persistent sein muss

**Das Grösste. Es ist keine Kleinigkeit an der Oberfläche, sondern die Frage, was ein Volumen
überhaupt ist.**

### Wie es sein soll

- Ein Volumen hat eine **gespeicherte** Ausdehnung, auf allen drei Achsen.
- Es wächst **nur durch explizite Handlungen**: einen Knoten anlegen, eine Zelle einfügen,
  eine Zeile einfügen. Das ist derselbe Satz, den `clamp_to_volume` für den Caret schon sagt —
  *„widening it is an explicit action, never a side effect of moving"* — nur gilt er dort für
  den Caret und sonst für nichts.
- **Bumping bleibt**: ein Edit darf kaskadierend verschieben, was im Weg steht. Aber die
  Kaskade endet **an der Volumengrenze** und schiebt nicht durch sie hindurch.

### Wie es ist

`LayoutGraph::grid_bounds()` (`src/layout.rs:2758`) **leitet** die Ausdehnung ab:

```rust
let mut max = IVec3::new(self.reserved_max.x, self.reserved_max.y, sink_z);
for id in self.layout_nodes.keys() {
    let Some(fp) = self.node_footprint(id) else { continue };
    max.x = max.x.max(fp.max.x);
    max.y = max.y.max(fp.max.y);
}
```

Daraus folgt dreierlei, und jedes davon ist die Abweichung:

- **`reserved_max` ist eine Untergrenze, keine Schranke.** Es merkt sich nur, was ein expliziter
  Insert geöffnet hat, und begrenzt nichts. Ein Knoten, dessen Footprint darüber hinausreicht,
  weitet das Volumen stillschweigend.
- **Z ist überhaupt nicht gespeichert**, sondern wird aus der Z des Sinks gelesen. Die Tiefe des
  Volumens ist damit eine Eigenschaft eines Knotens und nicht des Volumens.
- **Der Verdrängungslauf weitet als Nebenwirkung.** `move_node_delta` (`src/layout.rs:1362`)
  klemmt bei 0 und verweigert Züge auf die Quellreihe und auf die Sink-Reihe, aber auf der
  *fernen* X/Y-Seite begrenzt ihn nichts. Ein von `settle_footprints` hinausgedrängter Knoten
  vergrössert also das Volumen, ohne dass jemand darum gebeten hätte.
- Der Doc-Kommentar zu `clamp_to_volume` (`src/layout.rs:2782`) sagt es selbst, nur als
  Feststellung statt als Befund: *„the volume follows the nodes"*. Genau das soll es nicht.

### Zu tun

- [ ] `reserved_max: IVec2` → eine gespeicherte Ausdehnung auf allen drei Achsen (Vorschlag:
      `extent: IVec3`, inklusives Maximum; `min` bleibt der Ursprung und ist ohnehin fix).
- [ ] `grid_bounds()` gibt diese Ausdehnung **zurück**, statt sie über die Footprints zu maximieren.
- [ ] Z aus der Ausdehnung lesen statt aus dem Sink. Der Sink *sitzt* dann auf der letzten Reihe,
      statt sie zu *definieren* — `settle_sink` zieht ihn dorthin, weitet aber nicht mehr.
- [ ] Jede Stelle, die heute implizit weitet, explizit machen: Knoten anlegen, `plus_empty_cell`,
      `plus_empty_slab`, `plus_arm_row`. Die drei Insert-Wege schreiben `extent` ohnehin schon
      teilweise fort — sie müssen es künftig für alle drei Achsen und als einzige tun.
- [ ] `move_node_delta` gegen die Ausdehnung klemmen, nicht nur gegen den Ursprung.
- [ ] **Entscheiden und festhalten: was passiert, wenn eine Kaskade nirgendwo hin kann?**
      Der Plan wird heute schon als Ganzes verworfen, wenn ein Zug über den Ursprung hinausführt
      (`src/layout.rs:1422`) — dieselbe Behandlung an der fernen Grenze ist naheliegend und
      konsistent: **den Edit verweigern**, still, so wie die Verweigerung am Ursprung still ist.
      Die Alternative — automatisch weiten — ist genau der jetzige Zustand und fällt damit weg.
- [ ] Danach prüfen, ob eine Kante eines Knotens, der beim Typwechsel wächst (Anker werden höher,
      wenn ein Summentyp ankommt), noch Platz findet. Das ist der eine Fall, in dem heute etwas
      wächst, **ohne** dass der Benutzer eine Ausdehnung verlangt hat, und er braucht eine Antwort:
      weiten (dann ist es eine vierte explizite Handlung) oder verweigern (dann ist das Verdrahten
      der Kante der verweigerte Edit).

---

## 2. Zyklen werden gemeldet, nicht verhindert

Kap. 3.1.3 führt *no cycles* unter den **strukturellen Invarianten** und sagt: *„The graph stays
valid at all times, including while it is being edited."* Eine strukturelle Invariante ist etwas,
das der Editor durchhält, nicht etwas, das ein Linter anschliessend feststellt.

- `LayoutGraph::plus_edge` (`src/layout.rs:2031`) prüft **nur** die Ein-Kante-pro-Eingang-Regel.
  Ein Zyklus ist verdrahtbar.
- Abgefangen wird er dreimal *danach*: `infer::anchor_type_guarded` (`src/infer.rs:719`) gibt mit
  einem `visiting`-Set `Pending` zurück, `lint::cyclic_nodes` (`src/lint.rs:582`) meldet E6, und
  der Evaluator hat **gar kein** solches Set und würde rekursieren, bis der Stack weg ist — der
  Kommentar nennt E6 deshalb *„the one diagnostic that prevents a crash rather than a confusion."*

### Zu tun

- [ ] Erreichbarkeitsprüfung in `plus_edge`: führt `from` über bestehende Kanten (über
      Volumengrenzen hinweg, auf dem geflatteten Graphen) auf `to` zurück, wird die Kante nicht
      gezogen. Dieselbe Tür, an der schon die Ein-Kante-Regel steht.
- [ ] `E6` bleibt trotzdem stehen — als Netz gegen einen Fehler im Editor, nicht gegen die Sprache.
      Der `visiting`-Guard in `infer.rs` ebenso.
- [ ] Danach ist die Begründung in Kap. 4 die richtige herum: der Guard sichert gegen einen
      Implementierungsfehler, nicht gegen den Entwurf.

---

## 3. Die Z-Monotonie der Anordnung wird nicht erzwungen

Kap. 3.2.1: *„A node lies at a higher Z than every node whose result it consumes, directly or
indirectly."* Das steht dort als Regel, die ein Layout **erfüllen muss**.

Der **Kantenkurs** ist monoton — `EdgeCurve::from_endpoints` (`src/edge.rs:66`) klemmt die
Griff-Länge auf `l ≤ |dz|`, mit Beweis im Kommentar. Die **Knotenanordnung** ist es nicht: nichts
hindert daran, einen Konsumenten vor seinen Erzeuger zu ziehen. Die Kante läuft dann zwar immer
noch monoton, aber rückwärts, und das Bild behauptet eine Auswertungsreihenfolge, die es nicht gibt.

### Zu tun

- [ ] Entscheiden, **wo** die Regel greift, und dann nur dort:
      (a) beim Verdrahten — eine Kante, die gegen den Fluss liefe, wird nicht gezogen; oder
      (b) beim Verschieben — ein Zug, der einen Knoten hinter einen seiner Erzeuger brächte,
      wird verweigert; oder (c) beides.
      Vermutlich beides, mit derselben Prüfung an zwei Türen — es ist dieselbe Frage.
- [ ] Zusammen mit (1) denken: beide sind Verweigerungen an der Tür, und beide brauchen dieselbe
      Antwort darauf, wie eine stille Verweigerung dem Benutzer mitgeteilt wird. Heute ist eine
      verweigerte Verschiebung still (`src/layout.rs:1422`, *„Silently, because it is a refusal
      and not an anomaly"*) — das trägt, solange es selten ist, und wird fragwürdig, wenn es drei
      Regeln sind.

---

## 4. Die Lücke am Anker bei Typfehlschlag fehlt

Kap. 3.2.3, *Connected anchors of different types*: findet ein Fall des Ausgangsankers **keinen**
Fall am Eingangsanker, dann *„the edge keeps its shape and its Y level, but stops at the near
boundary of the anchor's cell instead of reaching it, so a gap stands where a connection would be."*

Der Code zielt stattdessen auf Zeile 0 (`src/main.rs:6810`):

```rust
// No matching leaf at the target: aim at its first row.
None => 0,
```

Beim `Match` wird die Lücke gezeichnet, und der Code **nennt den Unterschied selbst**
(`src/main.rs:6947`): *„Note the deliberate divergence from the edge pass, which aims an unmatched
leaf at row 0: docking a `Bool` arm onto an `Integer` band would draw the lie that it consumes it.
Here the gap is the statement."* Das Argument gilt für eine gewöhnliche Kante genauso.

### Zu tun

- [ ] Im Kantenlauf denselben Weg gehen wie `spawn_declared_cell_links`: kein Ziel gefunden →
      Strang bis an die *nahe* Grenze der Ankerzelle zeichnen und dort enden, statt auf Zeile 0
      anzudocken.
- [ ] Dabei auf den Unterschied achten, den 3.2.3 macht: die Lücke ist ein **Typfehler** (durchgezogen,
      mit Lücke), das gestrichelte Band ist **`pending`** (noch nichts bekannt). Beide zeigen eine
      Lücke, die Strichelung trägt, welcher Fall es ist.

---

## 5. Der Subtyp-Übergang wird als Zeilenwechsel gezeichnet, nicht als Verbreiterung

Kap. 3.2.3: *„Where a literal type meets its base type, the thin line broadens into the full band
over the course of the edge, and several literal types can merge into the same band."*

Eine gewöhnliche Kante wird mit **konstanter** Höhe und konstantem Linienmodus gebaut
(`src/main.rs:6828`, `height_start == height_end`, `line_mode_start == line_mode_end`). Das
Zusammenführen entsteht nur dadurch, dass mehrere Stränge auf **derselben Zielzeile** landen
(gefunden über `infer::subsumes`). Echtes Verjüngen und Öffnen gibt es ausschliesslich auf den
Struktur-Verbindungen von `Match` und `TypeCast`.

Der Shader kann es bereits — `edge_band.wgsl` rampt die Breite in Weltmassen über die Bogenlänge,
und `build_tapered_ribbon_mesh` baut die passende Geometrie. Es wird im Kantenlauf nur nicht benutzt.

### Zu tun

- [ ] Im Kantenlauf `build_tapered_ribbon_mesh` benutzen, wenn Quellzeile und Zielzeile
      unterschiedliche Formen haben (Linie → Band), und `line_mode_start`/`line_mode_end`
      entsprechend besetzen.
- [ ] Prüfen, was das für den Fall *mehrere Literale in dasselbe Band* heisst: laufen dann zwei
      sich öffnende Stränge übereinander? Falls ja, ist entweder die Regel in 3.2.3 gemeint als
      „sie treffen sich am Band und öffnen gemeinsam", oder sie braucht dort einen Halbsatz.
      **Erst am Bild entscheiden, dann schreiben.**

---

## 6. [!] Das Layout ist möglicherweise nicht kanonisch

Trifft eine Behauptung der Arbeit, nicht nur ein Bild: 3.2.2 sagt *„The same expression therefore
always has the same semantic structure"*, und die Versionskontroll-Geschichte in Kap. 6 hängt daran,
dass zwei visuell identische Programme nicht in der Datei differieren können.

`settle_footprints` (`src/layout.rs:1601`) iteriert `layout_nodes.keys()` **unsortiert** —
für `match_ids` (`:1615`), für `owner_ids` (`:1623`) — und wählt den Eindringling per `find_map`
über dieselbe ungeordnete Map (`:1662`). Jeder Besitzer verändert das gemeinsame Layout, und ein
späterer kann einen früheren erneut stören. Das Ergebnis kann damit im Prinzip von der
Hash-Iterationsreihenfolge abhängen.

Dass es ein Versehen ist und keine Haltung, zeigt der Kontrast: `lint.rs` sortiert seine Ids,
`source_order` sortiert mit `total_cmp` samt Id-Tiebreak, `hop_candidates`, `reachable_anchors`
und `create_candidates` sortieren ebenfalls — und jede dieser Stellen begründet es im Kommentar.

### Zu tun

- [ ] `match_ids` und `owner_ids` sortieren (nach `node::Id`, das ist bereits `Ord`).
- [ ] Die Eindringling-Auswahl deterministisch machen: statt `find_map` über die Map die
      Kandidaten sammeln, nach Adresse und Id sortieren und den ersten nehmen — dieselbe Form,
      die `hop_candidates` schon hat.
- [ ] Danach eine Gegenprobe: dasselbe Programm zweimal aus demselben Ablauf bauen und die
      Positionen vergleichen. Solange es keine Serialisierung gibt, reicht ein `Vec` der
      `(node::Id, IVec3)` sortiert und verglichen.

---

## 7. [!] `LayoutNode::pos` ist `Vec3`, sollte aber eine Zelladresse sein

Aufgefallen am 23.09.\ beim Zeichnen der Datenstruktur für Abbildung 4.2: der Autor ging davon aus,
dass die Position ganzzahlig ist. Sie ist es nicht.

```rust
// src/layout.rs:100
pub struct LayoutNode {
    pub node_id: crate::model::node::Id,
    pub pos: Vec3,          // <- f32 x 3
    pub shape: NodeShape,   // <- Vec<(IVec3, CellRole)>, also ganzzahlig
    pub gap_above: i32,
}
```

**Der Widerspruch steht im selben `struct`:** `shape` adressiert Zellen mit `IVec3`, `pos` mit
`Vec3`. Jede Stelle, die aus `pos` eine Adresse braucht, rundet — `ln.pos.round().as_ivec3()`
kommt im Layout und in `main.rs` mehrfach vor.

**Warum das mehr als Kosmetik ist.** 3.2.2 sagt, jede Koordinate sei ein nicht-negativer
*ganzzahliger* Wert und eine Zelle habe eine *Adresse*. Eine Fliesskommazahl ist keine Adresse:
sie kann zwei Werte haben, die auf dieselbe Zelle runden, und sie macht Gleichheit und damit auch
die Kanonizität des Layouts (§6) wackliger, als sie sein müsste.

**Kein Grund dagegen gefunden:** die Animation läuft über `Transform` der Bevy-Entities
(`animate_nodes`, `src/main.rs:6054`) und nicht über `pos`, also braucht die Zwischenwerte niemand.

### Zu tun

- [ ] `pos: Vec3` → `pos: IVec3`, und die `round().as_ivec3()`-Aufrufe entfallen lassen.
- [ ] Dabei prüfen, ob irgendwo eine Zwischenposition gebraucht wird — falls ja, gehört sie in die
      Darstellung und nicht in das Layout.
- [ ] Zusammen mit §6 erledigen: ganzzahlige Positionen sind die halbe Miete für ein vergleichbares,
      kanonisches Layout.
- [ ] Danach `figures/implementation-architecture.tex` nachziehen --- dort steht heute `Vec3`, mit
      einem Kommentar, der auf diesen Punkt zeigt.

---

## 8. Kleinkram, aufgelaufen beim Gegenlesen

- [ ] `EType::SumType` vergleicht in `types_match` (`src/infer.rs:1231`) **paarweise in
      Reihenfolge**, also sind `Integer|None` und `None|Integer` nicht gleich. In der Praxis
      baut nur `or_none` und `normalize_leaves` Summen, also ist die Ordnung faktisch kanonisch —
      aber es ist eine Falle, die beim nächsten Aufrufer zuschnappt.
- [ ] `FunctionDeclarationId` ist der Zeilenindex im Katalog (`src/model/function_declaration.rs:31`),
      also nur unter Anhängen stabil. Wird irgendwann serialisiert, muss es ein Name sein.
- [ ] **Das `E`-Präfix ist keine Konvention, sondern eine Familie — entscheiden, ob das so
      bleiben soll.** Gezählt am 2026-09-23: von 31 Enums tragen **fünf** ein `E`-Präfix —
      `ENode` (`src/model/node.rs:21`), `EAnchor` (`src/model/anchor.rs:20`), `EType`
      **zweimal** (`src/model/type.rs:11` und `src/infer.rs:14`) und `EValue`
      (`src/eval.rs:20`). Alle fünf benennen ein Ding der **Sprache**. Die übrigen 26 tragen
      keines: `CellRole`, `CastKind`, `Severity`, `CameraMode`, `Axis`, `PickKind` und so fort.
      (`EditorMode`, `EditTarget`, `EvalPhase` beginnen zwar mit `E`, aber als Wort — *Editor*,
      *Edit*, *Eval* —, nicht als Präfix.)
      **Gelesen als „Enums heissen `E…`" ist der Code an 26 von 31 Stellen falsch; gelesen als
      „Typen der Sprache heissen `E…`" ist er vollständig konsistent.** Die zweite Lesart ist
      die wahrscheinliche und die nützlichere, weil sie eine Schicht markiert statt einer
      Sprachkonstruktion. Unter ihr wäre `ECellRole` **falsch**: eine Zellenrolle ist ein Ding
      des Layouts, kein Ding der Sprache.
      Aufgekommen beim Zeichnen von Abbildung 4.2 der Arbeit
      (`../bsc-thesis/figures/implementation-datamodel.tex`), die deshalb `CellRole` schreibt.
      Wird anders entschieden, sind beide Stellen zu ändern.
      **Der eine Punkt, der unabhängig von der Entscheidung stört:** `EType` gibt es zweimal,
      in `model::r#type` und in `infer`, mit verschiedenen Varianten und einer Einbahn-Brücke
      (`infer::graph_type_to_eval_type`). Zwei gleichnamige Typen in einem Crate sind eine
      Falle, ganz gleich wie das Präfix ausgeht.
- [ ] Die drei harten Schleifendeckel (`0..128` in `settle_footprints` und in `move_node_delta`,
      `MAX_WALK_STEPS` beim Strahl) stehen begründet da und sollen bleiben — aber wenn (1) das
      Volumen zu einer echten Schranke macht, ist zu prüfen, ob der Deckel in `settle_footprints`
      danach überhaupt noch erreicht werden kann.
