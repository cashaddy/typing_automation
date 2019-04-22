/* Check and mark the validity of the test sections a */

ALTER TABLE test_sections ADD COLUMN is_valid bool DEFAULT 1, ALGORITHM=INSTANT;

/* Mark test sections with undefined keys as invalid */
UPDATE test_sections ts
SET ts.is_valid = 0
WHERE ts.TEST_SECTION_ID IN(
	SELECT TEST_SECTION_ID FROM(
		SELECT TEST_SECTION_ID FROM log
		WHERE log.DATA = 'undefined'
		GROUP BY log.TEST_SECTION_ID HAVING COUNT(DISTINCT(log.DATA)) = 1
	) AS ts_invalid
);

/* Mark test sections which are not in the log as invalid */
UPDATE test_sections ts 
SET ts.is_valid = 0
WHERE ts.TEST_SECTION_ID NOT IN
(
	SELECT DISTINCT(TEST_SECTION_ID) FROM log l
)