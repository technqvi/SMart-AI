create or replace view public.pmr_project
            (project_id, enq, project_name, project_start, project_end, company_id, company, has_pm,updated_at) as
SELECT app_project.id                                  AS project_id,
       app_project.enq_id                              AS enq,
       app_project.project_name,
       app_project.project_start,
       app_project.project_end,
       app_project.company_id,
       (SELECT app_company.company_name
        FROM app_company
        WHERE app_project.company_id = app_company.id) AS company,
       app_project.has_pm,app_project.updated_at
FROM app_project
WHERE app_project.is_dummy = false;

alter table public.pmr_project
    owner to postgres;

