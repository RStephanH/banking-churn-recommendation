"""
Column name mapping for the Santander Product Recommendation dataset.

The raw dataset uses Spanish column names. This maps them to English,
based on Kaggle's official data dictionary for this competition.

Usage:
    import pandas as pd
    from columns import COLUMN_MAPPING, PRODUCT_COLUMNS

    df = pd.read_csv("train_sampled_75k.csv")
    df = df.rename(columns=COLUMN_MAPPING)
"""

COLUMN_MAPPING = {
    "fecha_dato": "snapshot_date",
    "ncodpers": "customer_id",
    "ind_empleado": "employee_index",
    "pais_residencia": "country_residence",
    "sexo": "sex",
    "age": "age",
    "fecha_alta": "first_contract_date",
    "ind_nuevo": "new_customer_index",
    "antiguedad": "seniority_months",
    "indrel": "primary_customer_index",
    "ult_fec_cli_1t": "last_primary_date",
    "indrel_1mes": "customer_type_month_start",
    "tiprel_1mes": "relation_type_month_start",
    "indresi": "residence_index",
    "indext": "foreigner_index",
    "conyuemp": "spouse_employee_index",
    "canal_entrada": "join_channel",
    "indfall": "deceased_index",
    "tipodom": "address_type",
    "cod_prov": "province_code",
    "nomprov": "province_name",
    "ind_actividad_cliente": "activity_index",
    "renta": "gross_income",
    "segmento": "segment",
    # Product indicator columns (24 total) -- these form the
    # client-product matrix used by the recommender.
    "ind_ahor_fin_ult1": "product_saving_account",
    "ind_aval_fin_ult1": "product_guarantees",
    "ind_cco_fin_ult1": "product_current_account",
    "ind_cder_fin_ult1": "product_derivada_account",
    "ind_cno_fin_ult1": "product_payroll_account",
    "ind_ctju_fin_ult1": "product_junior_account",
    "ind_ctma_fin_ult1": "product_mas_particular_account",
    "ind_ctop_fin_ult1": "product_particular_account",
    "ind_ctpp_fin_ult1": "product_particular_plus_account",
    "ind_deco_fin_ult1": "product_short_term_deposit",
    "ind_deme_fin_ult1": "product_medium_term_deposit",
    "ind_dela_fin_ult1": "product_long_term_deposit",
    "ind_ecue_fin_ult1": "product_e_account",
    "ind_fond_fin_ult1": "product_funds",
    "ind_hip_fin_ult1": "product_mortgage",
    "ind_plan_fin_ult1": "product_pension_plan",
    "ind_pres_fin_ult1": "product_loans",
    "ind_reca_fin_ult1": "product_taxes",
    "ind_tjcr_fin_ult1": "product_credit_card",
    "ind_valo_fin_ult1": "product_securities",
    "ind_viv_fin_ult1": "product_home_account",
    "ind_nomina_ult1": "product_payroll",
    "ind_nom_pens_ult1": "product_pension_payment",
    "ind_recibo_ult1": "product_direct_debit",
}

# Convenience list: the English names of the 24 product columns,
# useful for isolating the client-product matrix from customer
# attribute columns.
PRODUCT_COLUMNS = [v for k, v in COLUMN_MAPPING.items() if v.startswith("product_")]
