/*
Select the log representing 10000 random participants which have only valid test sections.
*/

/* renamed ite_auto_new, ite_pred_new, ite_swyp_new */

CREATE TABLE IF NOT EXISTS log_sample
SELECT l.TEST_SECTION_ID,l.DATA,l.INPUT_LEN,l.INPUT_LDIST,l.ITE_AUTO,l.ITE_PRED,l.ITE_SWYP,l.INPUT,l.TIMESTAMP FROM log l
INNER JOIN
(
	SELECT ts.TEST_SECTION_ID FROM test_sections ts
	INNER JOIN
	(
		SELECT PARTICIPANT_ID FROM test_sections ts
		WHERE ts.is_valid = 1
		GROUP BY ts.PARTICIPANT_ID HAVING COUNT(*) = 15 ORDER BY RAND(42) LIMIT 35000
	) AS p
	ON ts.PARTICIPANT_ID = p.PARTICIPANT_ID
) AS ts2
ON l.TEST_SECTION_ID = ts2.TEST_SECTION_ID