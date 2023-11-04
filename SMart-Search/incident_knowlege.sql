select
incident.id as id,incident.incident_no as incident_no
,status.incident_status_name as status,severity.severity_name as  severity
,incident.incident_subject as subject,incident.incident_description as description
,TO_CHAR(incident.incident_datetime  AT TIME ZONE 'Asia/Bangkok','YYYY-MM-DD HH24:MI') as open_datetime
,TO_CHAR(incident.incident_close_datetime  AT TIME ZONE 'Asia/Bangkok','YYYY-MM-DD HH24:MI') as close_datetime
,xtype.incident_type_name as incident_type,service.service_type_name service_type
,inventory.serial_number
-- ,service_level.sla_name as sla
,product_type.productype_name as product_type,brand.brand_name as brand,model.model_name as model


from app_incident as incident
inner join app_incident_type as  xtype on incident.incident_type_id = xtype.id
inner join  app_incident_status as status on incident.incident_status_id = status.id
inner join  app_incident_severity as severity on  incident.incident_severity_id = severity.id
inner join  app_service_type as service on incident.service_type_id= service.id
inner join app_inventory as inventory on incident.inventory_id = inventory.id
inner join app_brand as brand on inventory.brand_id = brand.id
inner join app_model as model on inventory.model_id = model.id
inner join app_product_type as product_type on inventory.product_type_id = product_type.id
inner join app_sla as service_level on inventory.customer_sla_id = service_level.id

-- inner join app_project as project on inventory.project_id = project.id
-- inner join app_company as company on project.company_id = company.id

where incident_datetime>='2023-01-01'
and incident.incident_status_id=4
and inventory.is_dummy=False
limit 10



select  detail.id, detail.incident_master_id as incident_id ,
workaround_resolution as resolution,
TO_CHAR(task_start,'YYYY-MM-DD HH24:MI:SS') as task_start,
TO_CHAR(task_end,'YYYY-MM-DD HH24:MI:SS') as task_end,
team.service_team_name
from app_incident_detail detail
inner join  app_serviceteam team on detail.service_team_id=team.id
inner  join  app_employee engineer on detail.employee_id=engineer.id
 where detail.incident_master_id = 2360