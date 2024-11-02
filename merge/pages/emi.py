import streamlit as st

import plotly.express as px
# import plotly.graph_objects as go

st.title("Home :red[Loan] EMI :blue[Calculator] ", )
st.page_link("pages/res.py", label="LLM_response", icon="🏠")
st.page_link("pages/restate.py", label="Chat_history_response", icon="🏠", )
st.markdown("Calculate the EMI that you will be required to pay for your home loan with our easy-to-use home loan EMI calculator.")

def EMI(d, p, r, n):  
        # st.info("EMI = P x R x (1+R)^N / [(1+R)^N-1]")
        try:
            # Convert inputs to appropriate types
            # p = int(p)
            r = float(r)
            # n = int(n)

            # Calculate the effective loan amount after down payment
            p = p - d

            # Calculate interest rate per month
            r_mon = r / (12 * 100)

            # Calculate Equated Monthly Installment (EMI)
            emi = (p * r_mon * ((1 + r_mon)**n)/((1 + r_mon)**n - 1))
            # emi = p * r * ((1 + r)*n) / ((1 + r)*n - 1)
            
            col1, col2 = st.columns(2,vertical_alignment="center")
            with col1:
                st.info(f"Down_payment [d]: ₹{d:,}")
                st.info(f"Principal [p]: ₹{p:,}")
                st.info(f"Interest rate [r]: {r}%")
                st.info(f"Monthly interest rate: {r_mon:.4f}",) #help=" interest_rate / (Months * 100)")
                st.info(f"Loan tenure: {n} months, ({n / 12:.2f} years)")

            with col2:
                st.success(f"Monthly EMI  ₹{emi:,.2f}")
                st.success(f"Total Payable Amount ₹{emi*n :,.2f}")
                st.success(f" Interest [ total - Principal ] amount ₹{(emi*n)-p:,.2f}")
                # st.success(f"Total Interest: ₹{(emi * n) - p:,.2f}")
            # st.success(f"Loan Amount: ₹{p:,}, EMI for {n} months: ₹{emi:,.2f}")
            # st.success(f"Prinnnnnnnnipaallllllll looooooannnnnnnnnn Amount: ₹{p:,}, EMI for {n} months is {emi:,.2f}, the tooootall payable ₹{(emi*n):,.2f}, te interested amount be {(emi*n)-p :,.2f}   ")
            
            # Calculate and plot EMI breakdown
            principal_amount = p
            total_interest = (emi * n) - p
            total_payment = emi * n

            payment_data = {
            "Components": ["Principal_Amount", "Total_Interest", "Total_Payment"],
            "Values": [principal_amount, total_interest, total_payment],
            }
            fig = px.pie(
                payment_data,
                 values="Values", 
                 names="Components", 
                 title="EMI Payment Breakdown", 
                 hole=0.4, 
                color='Components',
                color_discrete_map={'Principal_Amount':'yellow',
                                 'Total_Interest':'cyan',
                                #  'Total_Interest':'royalblue',
                                 'Total_Payment':'darkblue'},
                # hover_name="Total_Interest",
                opacity=0.7,
                # hover_data=['total_interest']
            )
            # fig.update_traces(textposition='inside', textinfo='percent+Components')

            st.plotly_chart(fig, use_container_width=True)


            #  # Plot the payment breakdown as a pie chart
            # labels = ['Principal Amount', 'Total Interest', "Total Amount"]
            # values = [principal_amount, total_interest, total_payment]

            # fig = go.Figure(data=[go.Pie(labels=labels, values=values, hole=.3)])
            # fig.update_layout(title_text="Payment Breakdown")

            # # Display the chart
            # st.plotly_chart(fig, use_container_width=True,hoverinfo='label+percent', textinfo='value', textfont_size=20,)

        except ValueError:
            st.error("Please enter valid numeric values for Loan Amount, Interest Rate, and Number of Installments.")


# collect the inputs from the user                
# st.markdown("Calculate the EMI that you will be required to pay for your home loan with our easy to understand home loan EMI calculator.")

p = st.number_input("Total Loan Amount:", placeholder="Enter the loan amount in (in ₹)numbers", min_value=1000000, max_value=55000000, value=6500000, help="the ranges from lakhs to cores.")
d = st.number_input("Down Payment:", min_value=0, max_value=int(p), value=500000, help="Enter the down payment (in ₹).")
r = st.slider("Rate of Interest (% per annum):", min_value=0.5, max_value=15.0, value=8.75, step=0.25, help="Choose the Interest Rate as per your Bank preference")
n = st.number_input("Number of Installments (in months):", placeholder="Enter number of monthly installments", value=120)


# Calculate and display EMI if all inputs are valid
if d and p and r and n:
    EMI(d, p, r, n)
                     
                    # P: The principal loan amount 
                    # R: The monthly interest rate, which is the annual interest rate divided by 12 
                    # N: The loan tenure in months


