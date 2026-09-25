---
default: patch
---

A value a `match` or `if` binds is freed at its last use even when the match's value is used rather than returned, as in `r = match make() { Some(b) -> use(b) ... }`. The scrutinee and what the arms bound used to be held until the function returned, so three such matches in one function held six cells where they now hold none. Strings and values whose type is a type variable are freed at their last use too, where they used to wait for the function to return. `scarlet/internal` gains `cells_live`, the cells a process holds now, for debugging and for the tests that pin this.
