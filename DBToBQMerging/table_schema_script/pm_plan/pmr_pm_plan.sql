create view or replace pmr_pm_plan (pm_id, project_id, planned_date, ended_pm_date, pm_period, team_lead, updated_at) as
SELECT pm.id                            AS pm_id,
       pm.project_id,
       pm.planned_date,
       pm.ended_pm_date,
       pm.remark                        AS pm_period,
       (SELECT emp.employee_name
        FROM app_employee emp
        WHERE emp.id = pm.team_lead_id) AS team_lead,
       pm.updated_at
FROM app_preventivemaintenance pm;

alter table pmr_pm_plan
    owner to postgres;
