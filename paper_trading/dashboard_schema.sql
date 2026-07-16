-- paper_trading/dashboard_schema.sql
-- Run this once in the Supabase SQL editor (Dashboard → SQL Editor → New query).
--
-- Adds the one column the uploader needs that Lovable's prompt didn't include:
-- a unique mt5_ticket so the uploader can upsert open trades and later UPDATE
-- the same row when MT5 closes the position (SL/TP hit or signal reversal).

ALTER TABLE trades
  ADD COLUMN IF NOT EXISTS mt5_ticket BIGINT UNIQUE;

-- Sanity check: confirm the column exists.
-- SELECT column_name, data_type FROM information_schema.columns
--   WHERE table_name = 'trades' AND column_name = 'mt5_ticket';
