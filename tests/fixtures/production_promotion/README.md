# Production promotion acceptance fixture

`mpk.json` pins the genuine production YOLO26x Pose candidate from the original
failed promotion. The artifact and compile provenance remain unchanged. The
fixture temporarily restores only that profile's original on-demand target
requirements; it does not change model definitions, compile artifacts, or the
source-controlled automatic-build policy.

This is an operator fixture, not a model qualification or a general publication
workflow. Apply its guards in order. Do not publish while another Develop catalog
publication is in flight. Save the baseline outside Git; it may contain unrelated
catalog data. Keep the temporary policy in place only for the reviewed test window.

A promoted selection is server-owned. The ordinary publication API cannot
withdraw it or disable its on-demand model. Cleanup therefore requires the
repository lock and a database transaction scoped to the exact fixture promotion
UUID, candidate, target, model and profile. Require exactly one matching approval,
preserve the promotion history, audit the withdrawal, and release only the matching
`approved_selection` hold. Then restore the captured policy through the CAS snapshot
API. Do not delete candidates, runs or artifact objects. If the required operator
cleanup path or reviewer is unavailable, prepare the fixture without activating it.
