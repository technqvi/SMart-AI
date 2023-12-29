create or replace view public.pmr_pm_item
            (pm_item_id, pm_id, is_pm, pm_engineer, actual_date, document_engineer, document_date, inventory_id,
             is_complete,updated_at) as
SELECT pm_item.id                                             AS pm_item_id,
       pm_item.pm_master_id                                   AS pm_id,
       pm_item.is_pm,
       (SELECT app_employee.employee_name
        FROM app_employee
        WHERE app_employee.id = pm_item.pm_engineer_id)       AS pm_engineer,
       pm_item.actual_date,
       (SELECT app_employee.employee_name
        FROM app_employee
        WHERE app_employee.id = pm_item.document_engineer_id) AS document_engineer,
       pm_item.document_date,
       pm_item.inventory_id,
       CASE
           WHEN pm_item.actual_date IS NULL OR pm_item.document_date IS NULL OR pm_item.pm_engineer_id IS NULL OR
                pm_item.document_engineer_id IS NULL THEN false
           ELSE true
           END                                                AS is_complete
	,updated_at
FROM app_pm_inventory pm_item;

alter table public.pmr_pm_item
    owner to postgres;

