# TODO

## 1. Test restructured CLI

Run all combinations with and without `--demo` to verify the restructured paths and Config work end-to-end:

```bash
python -m src.main --demo                                    # eng B24 final demo
python -m src.main                                           # eng B24 final full optimization
python -m src.main --faculty ubf --demo                      # ubf B24 final demo
python -m src.main --semester G25 --demo                     # eng G25 final demo
python -m src.main --midterm --demo                          # eng B24 midterm demo
python -m src.main --faculty ubf                             # ubf B24 final full
```

Verify: schedules generated in `runs/`, analysis runs for eng/B24 only, no crashes on other combos.

## 2. Investigate analytics.py

The analysis module needs review after the restructuring:
- `read_fac_xlsxs()` signature changed — verify all callers pass the correct `exp_path` and `reference_path`
- `frequency_table()` and `frequency_table2()` signatures changed — confirm they are called correctly from `analysis()`
- Several functions had hardcoded paths that were updated — end-to-end test with `--analyze <N>` on an existing experiment
- The `generate_manuel_fac_()` function in `reports.py` also had its signature changed

## 3. Investigate G25 midterm infeasibility

**Symptom**: Solver returns INFEASIBLE for G25 midterms.

**Suspected cause**: The horizon (num_days * slots_per_day) is too small relative to the number of non-overlapping exams required within a department — specifically Makine Mühendisliği. The current hard no-overlap constraint for same department+year exams makes scheduling impossible when there are too many exams to fit.

**Potential solution — relaxed overlap for non-mandatory courses**:

Allow non-mandatory (elective) courses within the same department and year to overlap in time (and even share a room if capacity permits), under these conditions:

1. **Credit-based feasibility check**: For a given department+year, if total mandatory course credits already fill the student's expected load, then any non-mandatory course can be treated as an elective that no student *must* take simultaneously with another.
2. **Overlap rules for non-mandatory courses**:
   - Can overlap in time with other courses in the same dept+year
   - Can share a room if total student count fits within room capacity
   - Mandatory courses retain the hard no-overlap constraint
3. **Caveat — repeating students**: Students repeating courses from previous years could have conflicts even with "non-mandatory" courses from their original year. We don't have per-student enrollment data to detect this. Possible mitigations:
   - Treat it as a soft constraint (minimize but allow)
   - Add a configurable threshold (e.g., allow overlap only if course year differs by 2+)
   - Accept the limitation and document it

**Implementation notes**:
- Requires a `KREDI` (credit) column in course data — currently missing from B24 xlsx but referenced in `entities.py` (was removed as unused). Need to add it back to the data or compute from AKTS.
- Constraint change would be in `solver.py` section (6) — dept/year no-overlap. Split into mandatory (hard no-overlap) and non-mandatory (allowed overlap under conditions).
- The `pass_course_midterm` list in `entities.py` may also need updating for G25.

## 4. Verify semester naming and data inventory

**Question**: What semesters actually exist and what data do we have for each?

The naming convention is `B` = Bahar (Spring), `G` = Güz (Fall), followed by a two-digit year. An academic year spans two semesters (e.g., 2024-2025 = G24 + B25). Current data files reference B24 and G25, but it's unclear whether:

- B24 means Spring 2024 (part of the 2023-2024 academic year)
- G25 means Fall 2025 (part of the 2025-2026 academic year)
- Was there ever B25, G24, or B26 data? If so, where is it?
- Are B24 and G25 actually consecutive semesters from the same academic year, or from different years entirely?

**Action items**:
- Audit which semester datasets were ever collected or used
- Confirm the naming convention matches the actual academic calendar
- If other semesters exist (G24, B25, etc.), add them under the appropriate `data/{faculty}/{semester}/` folders
- Update `SEMESTERS` list in `src/config.py` if new semesters are added
