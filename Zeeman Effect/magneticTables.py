import pandas as pd
import dataframe_image as dfi


for x in range(3):
    
    filepath = f"CSV_Data/Magnetic_Fields/mag{x+1}.csv"
    df = pd.read_csv(filepath)

    df_horizontal = df.T.reset_index()
    df_horizontal.columns = ["Column"] + [f"Row{i+1}" for i in range(df.shape[0])]

 
    styled = df_horizontal.style.set_caption(f"Magnetic Field Measurements, Session {x+1}").set_properties(subset=["Column"], **{"font-weight": "bold"})  
    styled = styled.hide(axis="index").hide(axis="columns").format("{:.3f}", subset=df_horizontal.columns[1:]) 
                                

    dfi.export(styled, f"IMG/Tables/mag{x+1}.png")



styled = df_horizontal.style.set_caption("Horizontal Table") \
                             .set_properties(**{
                                 "text-align": "center",
                                 "padding": "6px"
                             })