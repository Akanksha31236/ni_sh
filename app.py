import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.cluster import KMeans

st.set_page_config(page_title="Customer Personality Prediction",page_icon="🗑️",layout="centered")
st.title("🗑️ Customer Personality Prediction")
st.subheader("Devide customers depends on her monthly spending")

st.divider()

df = pd.read_csv("Customer.csv")

with st.expander("View Dataset"):
    df

x = df[["Age","Income","Online_Spending"]]

cluster = st.slider("Select Number Of Clusters",min_value=2,max_value=6,value=3)

model = KMeans(n_clusters=cluster,random_state=42)

df["Cluster"]= model.fit_predict(x)
cluster_name={
    0:"Budget Customer",
    1:"Regular Customer",
    3:"Premium Customer",
    4:"VIP Customer",
    5:"Rich Customer",
    6:"Ultra Rich Customer"
}
df["Customer_type"] = df["Cluster"].map(cluster_name)
st.subheader("Cluster Center")
st.dataframe(df)
st.subheader("Customer type")
st.dataframe(df[["Age","Income","Online_Spending","Cluster","Customer_type"]])
st.subheader("Cluster center")
centers = pd.DataFrame(model.cluster_centers_,columns=["Age","Income","Online_Spending"])
st.dataframe(centers)
st.success(f"Inertia score:{model.inertia_:2f}")

st.subheader("Graph")
fig,ax = plt.subplots(figsize=(6,3))
scatter = ax.scatter(df["Income"],df["Online_Spending"],c=df["Cluster"],cmap="ocean",s=10)
ax.scatter(model.cluster_centers_[:,1],model.cluster_centers_[:,2],marker="^",color="red",s=30,label="centeroids")
ax.set_title("Customer Personality Prediction")
ax.set_xlabel("Anual Income")
ax.set_ylabel("Online Spending")
ax.grid(True)
ax.legend()
st.pyplot(fig)