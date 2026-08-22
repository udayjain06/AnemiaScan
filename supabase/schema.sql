-- Run this in the Supabase SQL editor before enabling the CHW dashboard.
-- Images and identifiers are intentionally excluded.
create table if not exists public.screenings (
  id uuid primary key,
  created_at timestamptz not null default now(),
  band text not null check (band in ('Normal', 'Mild Risk', 'Moderate Risk', 'Severe Risk')),
  method text not null,
  pallor_score double precision not null,
  erythema_index double precision not null,
  avg_r double precision not null,
  avg_g double precision not null,
  avg_b double precision not null,
  saturation double precision not null,
  value double precision not null
);

alter table public.screenings enable row level security;
-- Do not create anonymous read/write policies. The future CHW dashboard must
-- use authenticated users and ward-scoped policies before production use.
