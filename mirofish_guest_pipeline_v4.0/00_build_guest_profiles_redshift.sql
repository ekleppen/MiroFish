-- MiroFish hotel simulator: guest-profile features (version 1)
-- Run on Amazon Redshift.
-- Source: ml_data.sample_guest_tours, kipsu_prod_v4.customer_numbers,
--         kipsu_prod_v4.convo, fact.message
-- Unit: one valid, single-reservation tour (a pragmatic stay proxy).
-- All behavioral features are computed before a status check, or before
-- a DOCUMENTED PROXY cutoff when no status check occurred.
-- The proxy cutoff is NOT a historical send time.
-- Sub-brand isn't included here: its mapping source is not yet identified.
-- Sampling actual source rows retains company/stay-length combinations.

DROP TABLE IF EXISTS ml_data.mirofish_guest_profiles_v1;

CREATE TABLE ml_data.mirofish_guest_profiles_v1 AS
WITH eligible_tours AS (
    SELECT DISTINCT
        company_id,
        customer_number_id,
        tour_start_date::date AS tour_start_date,
        tour_end_date::date AS tour_end_date,
        reservation_ids,
        reservation_start_dates::date AS stay_start_date,
        reservation_end_dates::date AS stay_end_date
    FROM ml_data.sample_guest_tours
    WHERE company_id IS NOT NULL
      AND customer_number_id IS NOT NULL
      AND reservation_ids IS NOT NULL
      AND reservation_ids NOT LIKE '%,%'
      AND reservation_start_dates IS NOT NULL
      AND reservation_end_dates IS NOT NULL
      AND reservation_start_dates NOT LIKE '%,%'
      AND reservation_end_dates NOT LIKE '%,%'
      AND tour_start_date IS NOT NULL
      AND tour_end_date IS NOT NULL
),
tours AS (
    SELECT
        MD5(
            company_id::varchar || '|' || customer_number_id::varchar || '|' ||
            tour_start_date::varchar || '|' || tour_end_date::varchar || '|' ||
            reservation_ids
        ) AS source_profile_key,
        company_id,
        customer_number_id,
        tour_start_date,
        tour_end_date,
        stay_start_date,
        stay_end_date,
        DATEDIFF(day, stay_start_date, stay_end_date) AS stay_nights
    FROM eligible_tours
    WHERE stay_end_date > stay_start_date
),
messages AS (
    SELECT DISTINCT
        t.source_profile_key,
        fm.msg_id,
        fm.msg_timestamp,
        -- Work consistently in HOTEL-LOCAL time for both observed/proxy cutoffs.
        (fm.msg_date_local::varchar || ' ' ||
         fm.msg_time_local::varchar)::timestamp AS message_local_at,
        fm.was_from_guest,
        fm.was_from_staff,
        fm.was_welcome_msg,
        fm.was_welcome_msg_response,
        fm.was_status_check,
        fm.msg_text,
        fm.status_check_score,
        fm.status_check_scale
    FROM tours t
    JOIN kipsu_prod_v4.customer_numbers cn
      ON cn.id = t.customer_number_id
     AND cn.company_id = t.company_id
    JOIN kipsu_prod_v4.convo c
      ON c.company_id = cn.company_id
     AND c.sender = cn.number
    JOIN fact.message fm
      ON fm.company_id = t.company_id
     AND fm.convo_id = c.convo_id
     AND fm.msg_date_local BETWEEN t.tour_start_date AND t.tour_end_date
    WHERE fm.msg_id IS NOT NULL
      AND fm.msg_date_local IS NOT NULL
      AND fm.msg_time_local IS NOT NULL
),
ranked_checks AS (
    SELECT
        source_profile_key,
        message_local_at AS first_check_local_at,
        ROW_NUMBER() OVER (
            PARTITION BY source_profile_key
            ORDER BY msg_timestamp, msg_id
        ) AS rn
    FROM messages
    WHERE was_status_check = TRUE
),
first_checks AS (
    SELECT source_profile_key, first_check_local_at
    FROM ranked_checks
    WHERE rn = 1
),
cutoffs AS (
    SELECT
        t.*,
        CASE WHEN sc.source_profile_key IS NOT NULL THEN 1 ELSE 0 END
            AS has_historical_status_check,
        CASE WHEN sc.source_profile_key IS NOT NULL
             THEN 'observed_first_status_check'
             ELSE 'synthetic_reference_time' END AS cutoff_source,
        COALESCE(
            sc.first_check_local_at,
            CASE WHEN t.stay_nights = 1
                 THEN DATEADD(hour, 18, t.stay_start_date::timestamp)
                 ELSE DATEADD(hour, 12,
                                   DATEADD(day, 1, t.stay_start_date::timestamp))
            END
        ) AS profile_cutoff_at
    FROM tours t
    LEFT JOIN first_checks sc
      ON sc.source_profile_key = t.source_profile_key
),
pre_check_stats AS (
    SELECT
        c.source_profile_key,
        SUM(CASE WHEN m.was_from_guest = TRUE THEN 1 ELSE 0 END)
            AS prior_guest_messages,
        SUM(CASE WHEN m.was_from_staff = TRUE THEN 1 ELSE 0 END)
            AS prior_staff_messages,
        SUM(CASE WHEN m.was_welcome_msg = TRUE THEN 1 ELSE 0 END)
            AS prior_welcome_messages,
        SUM(CASE WHEN m.was_welcome_msg_response = TRUE
                  AND m.was_from_guest = TRUE THEN 1 ELSE 0 END)
            AS prior_welcome_responses,
        SUM(CASE WHEN m.was_from_guest = TRUE
                  AND COALESCE(m.was_welcome_msg_response, FALSE) = FALSE
                 THEN 1 ELSE 0 END)
            AS prior_other_guest_messages,
        SUM(CASE WHEN m.was_from_guest = TRUE
                  AND NULLIF(TRIM(m.msg_text), '') IS NOT NULL
                 THEN 1 ELSE 0 END)
            AS prior_guest_text_messages,
        AVG(CASE WHEN m.was_from_guest = TRUE
                  AND NULLIF(TRIM(m.msg_text), '') IS NOT NULL
                 THEN LEN(TRIM(m.msg_text))::decimal(18,2)
            END) AS avg_guest_message_chars
    FROM cutoffs c
    JOIN messages m
      ON m.source_profile_key = c.source_profile_key
     AND m.message_local_at < c.profile_cutoff_at
    GROUP BY c.source_profile_key
),
-- Score and scale must come from the SAME message: aggregating each with MAX()
-- separately can pair a 1-5 score with a 1-10 scale. Keep the first scored
-- message and count all scored messages so multi-score tours can be audited.
ranked_scores AS (
    SELECT
        source_profile_key,
        status_check_score,
        status_check_scale,
        ROW_NUMBER() OVER (
            PARTITION BY source_profile_key
            ORDER BY msg_timestamp, msg_id
        ) AS rn,
        COUNT(*) OVER (PARTITION BY source_profile_key) AS scored_message_count
    FROM messages
    WHERE status_check_score IS NOT NULL
),
scored_stays AS (
    SELECT
        source_profile_key,
        status_check_score,
        status_check_scale,
        scored_message_count
    FROM ranked_scores
    WHERE rn = 1
)
SELECT
    c.source_profile_key,
    c.company_id,
    c.stay_nights,
    CASE WHEN c.stay_nights > 30 THEN 1 ELSE 0 END AS long_stay_review_flag,
    c.has_historical_status_check,
    c.cutoff_source,
    c.profile_cutoff_at,
    sc.status_check_score,
    sc.status_check_scale,
    COALESCE(sc.scored_message_count, 0) AS scored_message_count,
    COALESCE(s.prior_guest_messages, 0) AS prior_guest_messages,
    COALESCE(s.prior_staff_messages, 0) AS prior_staff_messages,
    COALESCE(s.prior_welcome_messages, 0) AS prior_welcome_messages,
    COALESCE(s.prior_welcome_responses, 0) AS prior_welcome_responses,
    COALESCE(s.prior_other_guest_messages, 0) AS prior_other_guest_messages,
    COALESCE(s.prior_guest_text_messages, 0) AS prior_guest_text_messages,
    ROUND(s.avg_guest_message_chars, 1) AS avg_guest_message_chars
FROM cutoffs c
LEFT JOIN pre_check_stats s
  ON s.source_profile_key = c.source_profile_key
LEFT JOIN scored_stays sc
  ON sc.source_profile_key = c.source_profile_key;