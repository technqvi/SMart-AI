create or replace view public.pmr_inventory
            (inventory_id, serial_number, customer_warranty_start, customer_warranty_end, brand, model, product_type,
             project_id,updated_at) as
SELECT app_inventory.id                                            AS inventory_id,
       app_inventory.serial_number,
       app_inventory.customer_warranty_start,
       app_inventory.customer_warranty_end,
       (SELECT app_brand.brand_name
        FROM app_brand
        WHERE app_brand.id = app_inventory.brand_id)               AS brand,
       (SELECT app_model.model_name
        FROM app_model
        WHERE app_model.id = app_inventory.model_id)               AS model,
       (SELECT app_product_type.productype_name
        FROM app_product_type
        WHERE app_product_type.id = app_inventory.product_type_id) AS product_type,
       app_inventory.project_id,app_inventory.updated_at
FROM app_inventory
WHERE app_inventory.is_dummy = false;

alter table public.pmr_inventory
    owner to postgres;

