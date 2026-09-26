# TODOs

Stand 2026-09-22. Alles hier ist **Nachziehen an Kapitel 3 der Arbeit** — die Spezifikation
gilt, der Prototyp weicht ab. Die Arbeit geht durchgehend davon aus, dass es so implementiert
ist, wie Kapitel 3 es beschreibt; jede Abweichung unten ist also ein Fehler im Prototyp und
keine offene Entwurfsfrage.

Gefunden beim vollständigen Gegenlesen von Code gegen `03-concept-design.tex`
(siehe `../bsc-thesis/thesis-ch4-plan.md` §5 für die jeweilige Fundstelle in der Arbeit).

---

## 1. ~~[!] Das Volumen ist transient, wo es persistent sein muss~~ — erledigt

**Erledigt am 26.09.** Das Volumen ist jetzt eine gespeicherte Ausdehnung auf allen drei Achsen,
`grid_bounds()` gibt sie zurück statt sie abzuleiten, und sie wächst an genau zwei Stellen. Was
sich verschiebt, ist gegen sie geklemmt; was nirgendwo hin kann, wird verworfen. Die beiden
offenen Entscheidungen (1.6, 1.7) sind unten getroffen und begründet.

### Wie es sein soll

- Ein Volumen hat eine **gespeicherte** Ausdehnung, auf allen drei Achsen.
- Es wächst **nur durch explizite Handlungen**: einen Knoten anlegen, eine Zelle einfügen,
  eine Zeile einfügen. Das ist derselbe Satz, den `clamp_to_volume` für den Caret schon sagt —
  *„widening it is an explicit action, never a side effect of moving"* — nur gilt er dort für
  den Caret und sonst für nichts.
- **Bumping bleibt**: ein Edit darf kaskadierend verschieben, was im Weg steht. Aber die
  Kaskade endet **an der Volumengrenze** und schiebt nicht durch sie hindurch.

### Wie es jetzt ist

`LayoutGraph::extent: IVec3` ist die Ausdehnung, und `grid_bounds()` gibt sie zurück:

```rust
// src/layout.rs
pub fn grid_bounds(&self) -> Option<GridBounds> {
    self.sink_id()?;   // the one thing still asked of the nodes, and it is a
                       // yes-or-no question rather than a measure
    Some(GridBounds { min: IVec3::ZERO, max: self.extent })
}
```

Kein Knoten wird mehr gefragt, wie weit das Volumen reicht. Damit sind die drei Befunde
beantwortet:

- ~~**`extent` ist auf X/Y eine Untergrenze, keine Schranke.**~~ Sie ist jetzt auf allen drei
  Achsen die Schranke.
- ~~**Z ist überhaupt nicht gespeichert.**~~ Gespeichert, und `settle_sink` schreibt sie nicht mehr:
  der Sink wird auf die letzte Ebene gezogen, und ob das Volumen mitwachsen *darf*, ist die Frage
  des Edits und nicht die des Normalisierungslaufs.
- ~~**Der Verdrängungslauf weitet als Nebenwirkung.**~~ `move_node_delta` nimmt jetzt eine
  `Widening`-Politik: unter `Refused` ist es gegen beide Flächen geklemmt und verwirft einen Plan,
  der nicht hineinpasst, als Ganzes. Der Shove ist der einzige Aufrufer, der `Refused` verlangt.

**Zwei Türen, durch die das Volumen wächst, und keine dritte:**

1. `claiming_extent()` — wächst, um zu halten, was drinsteht. Bottom-up, nur nach aussen. Wird
   ausschliesslich von `GraphState::resettle()` gerufen, also von jedem Edit, der um Platz bittet.
2. Der Anspruch in `plus_empty_layer` — eine in leeren Raum geöffnete Ebene bewegt keinen Knoten,
   also ist dieser Anspruch der einzige Beleg, dass sie geöffnet wurde. `claiming_extent` kann
   messen, was sich bewegt *hat*; nicht, was sich nicht bewegen musste.

**Und zwei Prüfungen, denen ein Edit unterliegt, der nicht weiten darf** (`resettle_bounded`):
`within_extent()` und `footprints_settled()`. Die erste fängt die zwei Wege, auf denen eine
Verschiebung die ferne Fläche erreicht, ohne dorthin zu treten — ein Armstapel, der breiter
umpackt, und ein Sink, der nachgezogen wird. Die zweite fängt den Eindringling, den die Kaskade
nirgendwo hinschieben konnte: `settle_footprints` liest die unveränderte Position als „verweigert",
bricht ab und lässt ihn stehen. Zwei Knoten auf einer Zelle ist kein Layout, das man behält.

### Was sich dadurch am Bedienen ändert

Das ist keine offene Frage, sondern die Folge, und sie gehört aufgeschrieben, weil sie sich beim
ersten Ausprobieren wie ein Fehler anfühlen kann.

Der Caret war schon immer ans Volumen geklemmt (`clamp_to_volume`) — nur konnte er praktisch
überall hin, weil jeder Knoten das Volumen mitzog. Jetzt zieht keiner mehr, also gilt die Klemme
wirklich. Ein frisches Programm ist eine Zelle breit und eine hoch: um einen zweiten Knoten
**neben** den ersten zu setzen, wird erst eine Spalte geöffnet (`Return`), dann der Caret bewegt,
dann gebaut. Dasselbe für eine Zeile (`Shift+Return`).

Das ist genau der Satz aus „Wie es sein soll" — *eine Zelle einfügen, eine Zeile einfügen* —, nur
vorher nicht zu merken. Was **nicht** erst Platz braucht: einen Knoten anlegen, der von sich aus
breiter ist (ein Aufruf mit drei Eingängen beansprucht seine drei Spalten selbst), einen Arm
hinzufügen, einen Typ oder Namen setzen, eine Kante ziehen. Die weiten alle.

### Zu tun

- [x] `reserved_max: IVec2` → `extent: IVec3`, inklusives Maximum; `min` bleibt der Ursprung
      und ist ohnehin fix. Erledigt am 26.09.
- [x] `grid_bounds()` gibt diese Ausdehnung **zurück**, statt sie über die Footprints zu
      maximieren. Erledigt am 26.09., zusammen mit 1.4/1.5/1.6, weil es ohne sie Knoten ausserhalb
      ihres eigenen Volumens hinterlassen hätte.
- [x] Z aus der Ausdehnung lesen statt aus dem Sink. Der Sink *sitzt* jetzt auf der letzten Reihe,
      statt sie zu *definieren*. Erledigt am 26.09. — `grid_bounds` liest `extent.z` und fragt den
      Sink nur noch, *ob* es einen gibt; `plus_empty_layer`, `settle_sink` und
      `harmonize_match_sinks` schreiben `extent.z` mit, wenn sie den Sink bewegen.
- [x] Jede Stelle, die implizit weitete, explizit gemacht. Erledigt am 26.09., aber anders als
      hier vorgeschlagen: nicht jeder Insert-Weg schreibt `extent` selbst, sondern **eine** Funktion
      tut es für alle — `claiming_extent()`, gerufen von `GraphState::resettle()`. Damit ist die
      Antwort strukturell statt aufzählend: jeder andere Pfad *liest* die Ausdehnung und keiner
      schreibt sie, und das ist prüfbar (`grep`). `settle_sink` und `harmonize_match_sinks` haben
      ihre Zuweisungen verloren. Die eine Ausnahme ist `plus_empty_layer`, und sie ist notwendig:
      eine in leeren Raum geöffnete Ebene bewegt keinen Knoten, also gibt es nichts zu messen.
- [x] `move_node_delta` gegen die Ausdehnung klemmen, nicht nur gegen den Ursprung. Erledigt am
      26.09. Dabei zwei Dinge gefunden, die mehr als der Buchstabe des Punktes sind:
      **(a) Zellen statt Adresse.** Geklemmt wurde die *Position*. Am Ursprung geht das auf, weil
      jede Form von ihrer Ursprungszelle nach +X/+Y/+Z wächst; an der fernen Fläche nicht. Ein
      Aufruf legt seinen Körper über alle seine Eingänge, ein `Match` reicht bis hinter den
      tiefsten Zweig. Geklemmt wird jetzt gegen `node_footprint`, verschoben um den Zug.
      **(b) Die Sink-Reihe war auch nur gegen die Position geprüft.** `target_z >= sink_z` liess
      die *hintere* Zelle eines tiefen Knotens auf der Sink-Reihe landen; `settle_sink` zog den
      Sink dann nach hinten und weitete. Jetzt `target_z + span.z >= sink_z`.
- [x] **Entschieden am 26.09.: den Edit verweigern, still.** Wie vorgeschlagen, und an derselben
      Tür wie die Verweigerung am Ursprung — `outside_volume` prüft beide Flächen und verwirft den
      Plan als Ganzes.
      Beim Umsetzen kam heraus, dass die Tür allein nicht reicht, und das ist der Teil, der hier
      vorher fehlte: eine Verschiebung erreicht die ferne Fläche auch, **ohne dorthin zu treten**.
      Ein senkrechter Zug an einem Arm ist eine Änderung des Abstands darüber
      (`with_arm_gap_delta`), und der Stapel packt erst im Settle breiter um; und ein Eindringling,
      den die Kaskade nirgendwo hinschieben kann, bleibt einfach stehen, weil
      `settle_footprints` die unveränderte Position als „verweigert" liest und abbricht. Beides
      hinterlässt nichts, was am Plan zu sehen wäre. Deshalb prüft `resettle_bounded` **nach** dem
      Settle `within_extent()` und `footprints_settled()` und stellt sonst das Layout von vorher
      wieder her.
      Eine Abweichung von „still": die Verweigerung schreibt eine `debug!`-Zeile mit dem Grund.
      Auf dem Bildschirm sind eine Verweigerung und ein Fehler nicht zu unterscheiden — in beiden
      Fällen passiert nichts —, und das ist die eine Zeile, die sie auseinanderhält.
- [x] **Entschieden am 26.09.: weiten. Das Umformen eines Knotens ist die vierte explizite
      Handlung.** Betrifft drei Wege, nicht nur den einen: einen Typ deklarieren (Anker werden
      höher), einen Namen tippen (der Körper wird länger), eine Kante ziehen (der Zielanker nimmt
      die Höhe dessen an, was ankommt).
      **Begründung.** Ein Knoten ist so gross, wie sein Typ und sein Name ihn machen. Alle drei
      Handlungen sind ausdrückliche Handlungen *an einem Knoten* und keine Nebenwirkungen — was
      §1 ausschliesst, ist Wachstum, um das niemand gebeten hat, und keine davon ist das.
      Verweigern hätte drei Preise: eine Verdrahtung, die stumm nichts tut und dem Benutzer keinen
      Weg lässt zu sehen, warum; eine Ankerhöhe, die davon abhängt, wie viel Platz zufällig übrig
      war — also ein Bild, das über den Typ lügt; und eine Reihenfolge-Abhängigkeit, in der
      dasselbe Programm je nach Bauweise verdrahtbar oder nicht ist, was §6 (Kanonizität) direkt
      widerspricht.
      Im Code ist das `Widening::Allowed` und damit kein Sonderfall: alle drei gehen durch
      `resettle()`, wie das Anlegen eines Knotens auch.

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
`source_order` sortiert samt Id-Tiebreak, `hop_candidates`, `reachable_anchors`
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
dass die Position ganzzahlig ist. Sie war es nicht.

**Erledigt am 26.09.** Der Befund, wie er stand:

```rust
// src/layout.rs, vor dem 26.09.
pub struct LayoutNode {
    pub node_id: crate::model::node::Id,
    pub pos: Vec3,          // <- f32 x 3
    pub shape: NodeShape,   // <- Vec<(IVec3, CellRole)>, also ganzzahlig
    pub gap_above: i32,
}
```

`pos` ist jetzt `IVec3`, und `gap_above` bleibt genau da, wo es ist: eine Arm-Zeile wird von
`respace_match_patterns` *abgeleitet* (erste Zeile fest, dann je Arm `+ gap_above`
`+ branch_row_height`), also ist `pos.y` eines Patterns eine Antwort. Den erklärten Weissraum aus
dem Abstand zweier Zeilen zurückzulesen hiesse, die Frage aus der Antwort zu stellen — und ein
Zweig, der in die Lücke wächst, frisst sie dann auf und gibt sie beim Schrumpfen nicht zurück.

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

- [x] `pos: Vec3` → `pos: IVec3`, und die `round().as_ivec3()`-Aufrufe entfallen lassen.
      Erledigt am 26.09. Mitgegangen sind vier Toleranzvergleiche (`< 0.5` zweimal, `+ 0.001`,
      `length_squared() < 0.25`), das `total_cmp` in `source_order` — auf `i32` gibt es kein NaN,
      die Ordnung ist total, weil der Typ es ist —, `PATTERN_LOCAL_Z: f32` → `i32` und
      `Axis::unit`. `render::cell_center_world` nimmt jetzt eine `IVec3`: das ist der eine Ort, an
      dem aus einer Adresse ein Punkt wird, und fünf Aufrufstellen in `main.rs` haben dafür ihr
      `.as_vec3()` verloren. `LayoutAnchor::pos` ist weg — es war immer `Vec3::splat(1.0)` und
      wurde nirgends gelesen, also gerade keine Adresse.
- [x] Dabei prüfen, ob irgendwo eine Zwischenposition gebraucht wird: **nein.** Positionen werden
      ausschliesslich in `src/layout.rs` geschrieben, und `animate_nodes` (`src/main.rs`) ist
      auskommentiert. Die einzigen echten Halbzahlen sind Zell- und Volumen-Mittelpunkte, und die
      entstehen beide erst in der Darstellung (`cell_center_world`, `spawn_volume_surfaces`).
- [ ] Zusammen mit §6 erledigen: ganzzahlige Positionen sind die halbe Miete für ein vergleichbares,
      kanonisches Layout. Die Gegenprobe ist jetzt billig — ein `Vec<(node::Id, IVec3)>` sortiert
      und verglichen —, aber §6 selbst (die ungeordnete Iteration in `settle_footprints`) steht noch.
- [ ] Danach `figures/implementation-datamodel.tex` nachziehen — dort steht `pos & Vec3` und
      `reserved_max & IVec2`, beides jetzt falsch, und ein `[!]`-Block zeigt auf diesen Punkt.
      Nicht bloss zwei Wörter: die Kommentare in der Abbildung rechnen Geometrie an der Breite der
      Zelle `reserved_max`+`IVec2` vor, und `extent`+`IVec3` ist schmaler.

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
- [x] Die drei harten Schleifendeckel (`0..128` in `settle_footprints` und in `move_node_delta`,
      `MAX_WALK_STEPS` beim Strahl) stehen begründet da und bleiben. Nachgesehen am 26.09., nachdem
      §1 das Volumen zu einer echten Schranke gemacht hat: der Deckel in `settle_footprints` ist
      **weiterhin erreichbar**, aber seltener, weil eine verweigerte Verschiebung jetzt früher
      abbricht — die Schleife liest die unveränderte Position und bricht ab, statt zu pollen. Neu
      ist, dass ein nicht konvergierter Lauf nicht mehr stillschweigend durchgeht: unter
      `Widening::Refused` prüft `footprints_settled()` genau das und verwirft den Edit.
      **Was offen bleibt:** unter `Widening::Allowed` wird nicht geprüft. Ein Edit, der um Platz
      bittet, bekommt ihn, also hat die Kaskade dort immer ein Ziel — bis auf den Deckel selbst.
      Falls der je zuschlägt, steht das Layout mit einer Überlappung da, so wie vorher auch.
