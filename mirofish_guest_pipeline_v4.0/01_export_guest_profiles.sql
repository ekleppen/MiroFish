-- Run after 00_build_guest_profiles_redshift.sql has built ml_data.mirofish_guest_profiles_v1.
-- Export this SELECT as guest_profiles.csv.
-- Accounts are deduplicated per property BEFORE the join, so every profile row is kept.
-- Sanity check: the exported row count must equal
--   SELECT COUNT(*) FROM ml_data.mirofish_guest_profiles_v1;
WITH accounts AS (
    SELECT
        kipsu_company_id__c,
        sub_brand__c
    FROM salesforce.accounts
    WHERE kipsu_company_id__c IS NOT NULL
    QUALIFY ROW_NUMBER() OVER (
        PARTITION BY kipsu_company_id__c
        ORDER BY launch_date__c DESC NULLS LAST
    ) = 1
)
SELECT
    m.source_profile_key,
    m.company_id,
    a.sub_brand__c AS sub_brand,
    m.stay_nights,
    m.long_stay_review_flag,
    m.has_historical_status_check,
    m.cutoff_source,
    m.prior_guest_messages,
    m.prior_staff_messages,
    m.prior_welcome_messages,
    m.prior_welcome_responses,
    m.prior_other_guest_messages,
    m.prior_guest_text_messages,
    m.avg_guest_message_chars,
    m.status_check_score,
    m.status_check_scale,
    m.scored_message_count
FROM ml_data.mirofish_guest_profiles_v1 m
LEFT JOIN accounts a
  ON a.kipsu_company_id__c = m.company_id;
