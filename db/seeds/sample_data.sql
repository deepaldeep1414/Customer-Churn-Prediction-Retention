-- ============================================================
-- Migration: 003_seed_sample_data.sql
-- Description: Sample data for local development and testing.
--              DO NOT run in production.
-- ============================================================

-- Insert sample customers
INSERT INTO curated.customers (customer_id, external_id, email_hash, gender, senior_citizen, partner, dependents, signup_date)
VALUES
    ('a1b2c3d4-0001-0001-0001-000000000001', 'EXT-0001', sha256('alice@example.com'::bytea)::text, 'female', false, true,  true,  '2020-01-15'),
    ('a1b2c3d4-0002-0002-0002-000000000002', 'EXT-0002', sha256('bob@example.com'::bytea)::text,   'male',   false, false, false, '2021-06-01'),
    ('a1b2c3d4-0003-0003-0003-000000000003', 'EXT-0003', sha256('carol@example.com'::bytea)::text, 'female', true,  true,  false, '2022-03-20'),
    ('a1b2c3d4-0004-0004-0004-000000000004', 'EXT-0004', sha256('dave@example.com'::bytea)::text,  'male',   false, false, true,  '2019-11-05'),
    ('a1b2c3d4-0005-0005-0005-000000000005', 'EXT-0005', sha256('eve@example.com'::bytea)::text,   'female', false, true,  false, '2023-02-10')
ON CONFLICT DO NOTHING;

-- Insert subscriptions
INSERT INTO curated.subscriptions (customer_id, contract_type, start_date, end_date)
VALUES
    ('a1b2c3d4-0001-0001-0001-000000000001', 'two_year',        '2020-01-15', NULL),
    ('a1b2c3d4-0002-0002-0002-000000000002', 'month-to-month',  '2021-06-01', '2023-08-31'),
    ('a1b2c3d4-0003-0003-0003-000000000003', 'one_year',        '2022-03-20', NULL),
    ('a1b2c3d4-0004-0004-0004-000000000004', 'month-to-month',  '2019-11-05', NULL),
    ('a1b2c3d4-0005-0005-0005-000000000005', 'month-to-month',  '2023-02-10', NULL)
ON CONFLICT DO NOTHING;

-- Insert billing
INSERT INTO curated.billing (customer_id, billing_period_start, billing_period_end, monthly_charges, total_charges, payment_method, paperless_billing, days_overdue)
VALUES
    ('a1b2c3d4-0001-0001-0001-000000000001', '2023-06-01', '2023-06-30', 79.99, 3839.52, 'credit_card',       true,  0),
    ('a1b2c3d4-0001-0001-0001-000000000001', '2023-07-01', '2023-07-31', 79.99, 3919.51, 'credit_card',       true,  0),
    ('a1b2c3d4-0002-0002-0002-000000000002', '2023-06-01', '2023-06-30', 49.99, 1249.75, 'electronic_check',  false, 5),
    ('a1b2c3d4-0003-0003-0003-000000000003', '2023-06-01', '2023-06-30', 94.50, 1606.50, 'bank_transfer',     true,  0),
    ('a1b2c3d4-0004-0004-0004-000000000004', '2023-06-01', '2023-06-30', 29.99, 1409.53, 'mailed_check',      false, 12),
    ('a1b2c3d4-0005-0005-0005-000000000005', '2023-06-01', '2023-06-30', 65.00,  390.00, 'electronic_check',  true,  0)
ON CONFLICT DO NOTHING;

-- Insert services
INSERT INTO curated.services (customer_id, phone_service, internet_service, online_security, online_backup, device_protection, tech_support, streaming_tv, streaming_movies, effective_from)
VALUES
    ('a1b2c3d4-0001-0001-0001-000000000001', true, 'fiber_optic', true,  true,  true,  true,  true,  true,  '2020-01-15'),
    ('a1b2c3d4-0002-0002-0002-000000000002', true, 'dsl',         false, false, false, false, false, false, '2021-06-01'),
    ('a1b2c3d4-0003-0003-0003-000000000003', true, 'fiber_optic', true,  true,  false, true,  true,  false, '2022-03-20'),
    ('a1b2c3d4-0004-0004-0004-000000000004', true, 'none',        false, false, false, false, false, false, '2019-11-05'),
    ('a1b2c3d4-0005-0005-0005-000000000005', true, 'dsl',         false, true,  false, false, true,  true,  '2023-02-10')
ON CONFLICT DO NOTHING;

-- Insert support events
INSERT INTO curated.support_events (customer_id, ticket_category, ticket_status, opened_at, resolved_at, csat_score)
VALUES
    ('a1b2c3d4-0002-0002-0002-000000000002', 'billing',             'resolved', '2023-05-10 09:00:00+00', '2023-05-12 14:00:00+00', 2),
    ('a1b2c3d4-0002-0002-0002-000000000002', 'cancellation_intent', 'closed',   '2023-07-20 11:00:00+00', '2023-07-21 10:00:00+00', 1),
    ('a1b2c3d4-0004-0004-0004-000000000004', 'billing',             'resolved', '2023-06-05 08:00:00+00', '2023-06-07 16:00:00+00', 3),
    ('a1b2c3d4-0004-0004-0004-000000000004', 'technical',           'resolved', '2023-07-15 10:00:00+00', '2023-07-16 09:00:00+00', 4),
    ('a1b2c3d4-0005-0005-0005-000000000005', 'general',             'open',     '2023-07-28 13:00:00+00', NULL,                     NULL)
ON CONFLICT DO NOTHING;
