# 

2 Visualisierungstheorien
 - quantitativ/ordinal
 - nominal <<--- das was ich will
 


### GeckoGraph: A Visual Language for Polymorphic Types
https://arxiv.org/pdf/2405.12699

graphische notation for types

empirische study (714 participants)! largest on functional programming ever conducted!

same design goals:
(D1) Low barrier to learn.
(D2) Easy to parse for humans. 
(D3) Easy to compare and search. 

Primary encoding: color, secondary text fallback (first 2 letters)
- same color: gap/angled corner for big zoomout separation visible

2 dimensions -> type nesting grows taller, more arguments grows horizontally

wrapper style -> easy via mouse hover tell, what the user wants to be selected

previous iterations of Geckograph: 3-dimensional rendering dismissed because of too much effort and difficult readability on zoom out

decision against geometric symbols, because limited and quickly indistinguishable


evaluation via 10-level game "zero to hero": compose functions with their types to build a function from type "zero" to type "hero"

colors are important feature:
review article[27] solid color hue in  filled shape = stronger visual perception for nominal data

clear defined dimensions also very intuitive and strong

low barrier: respects left to right reading order, familiar symbolic name as secondary encoding




weaknesses: space usage vertical...
colors- color blind/impaired people ... when strong usage-> at least use color blind freandly schemes!!!

related work: Jung's notation [18]


7-9, 23 show that visual programming languages do help beginners with understanding


for me:
-------> easier types -> more freedom with visualization?







## rustviz
https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9833121
borrowchecker visualizer
color is a secondary notation


## McKinley
https://dl.acm.org/doi/epdf/10.1145/22949.22950

-> effectiveness of presentation used from cleveland and mcguill (studied accuracy of quantitativ graphical presentations)

for nominal:
1. position
2. color hue
3. texture
4. connection
5. containment
6. density 
7. color saturation  <- dont use...use for ordinal
8. shape
9. length   -> size perceived as ordinal
10. angle    XXX
11. slope    XXX
12. area    -> size
13. volume  -> size

[23]
parts of color spectrum can be perceived as ordinal

## Bertin
https://countersubject.biz/wp-content/uploads/2024/08/Semiology-of-Graphics-Diagrams-Networks-Maps-Jacques-Bertin-Z-Library.pdf
bertin: 6 retinal techniques


## graphical incremental type inference. a graph transformation approach
https://upcommons.upc.edu/server/api/core/bitstreams/560b4660-7b98-44d2-9a4b-91f8620925e5/content
NiMo system
https://upcommons.upc.edu/server/api/core/bitstreams/06bb9529-cb37-4416-803a-aacd59e52b8c/content
auch leichter shape aber viel color auch


## metastudie
A Review and Collation of Graphical Perception Knowledge for
Visualization Recommendation
