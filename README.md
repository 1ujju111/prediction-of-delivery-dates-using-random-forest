# prediction-of-delivery-dates-using-random-forest
The project employs a Random Forest Regressor to predict the Estimated Delivery Date (EDD). This algorithm is highly effective for regression problems due to its ability to handle non-linear relationships, feature interactions, and large datasets. 
Model Performance Metrics:
1.	Mean Squared Error (MSE):
o	Training data MSE: 0.08792
o	Testing data MSE: 0.548487
2.	R² (Coefficient of Determination):
o	Training data R²: 0.974763
o	Testing data R²: 0.842906

![image](https://github.com/user-attachments/assets/a2aa3d02-c45f-4ec1-a44a-77cc425182d5)
Above chart is scatter plot of predicted values by model vs given values in the dataset with a trendline of same.

These metrics indicate that the model has high accuracy on the training dataset, capturing 97.48% of the variance. On the testing dataset, it retains robust predictive performance with 84.29% variance explained, while maintaining a low error margin.
Target Variable: 
	• 	predicted_exact_sla: The predicted number of days between shipment and delivery. 
Features Used: 
1.	courier_partner_id: Identifies the courier partner. 
2.	account_type_id: Distinguishes account types, reflecting shipping agreements or priorities. 
3.	drop_pin_code and pickup_pin_code: Geographic information influencing transit time. 
4.	quantity: Total number of items in the shipment. 
5.	account_mode: Shipment mode (e.g., Air or Surface). 
6.	delivery_days: Calculated time difference between shipment and delivery dates. 
NAME: Ujjawal Chauhan 
(2K21/EN/47) 
 
 
Key Findings:                                                                                                            
1.	Delivery Time Summary: 
•	Average delivery time: ~2.49 days. 
•	Minimum delivery time: -35 days (indicating data errors or inconsistencies). 
•	Maximum delivery time: 54 days. 
 
2.	The analysis of shipment volumes by shipping mode reveals: 
•	Air: 1,202,664 shipments 
•	Default: 143,268 shipments 
•	Heavy Surface: 19,710 shipments 
•	Surface: 35,189 shipments 
  
 
 ![image](https://github.com/user-attachments/assets/868f6aea-86f0-4ced-bc0a-40788607e889)

 
3.	The comparison of average delivery times across shipping modes reveals: 
•	Air: ~2.69 days (fastest) 
•	Default: ~-0.01 days (possible data inconsistency) 
•	Heavy Surface: ~4.83 days 
•	Surface: ~4.43 days 
The "Air" mode delivers significantly faster on average compared to surfacebased modes. However, the "Default" mode shows an unusual negative average delivery time, suggesting potential data errors or inconsistencies. 
  ![image](https://github.com/user-attachments/assets/746592c6-88ae-4eff-9074-9a9750514d97)

4.	Pin Code with the Highest Average Delivery Time: 
•	Pin Code: 741403 
•	Average Delivery Time: 8 days 
•	SLA Compliance Rate: 100% 
 
5.	Pin Code with the Lowest Average Delivery Time: 
•	Pin Code: 998899 
•	Average Delivery Time: 0 days 
•	SLA Compliance Rate: 100% 
6.	District with the Highest Average Delivery Time: 
•	District: Khawzawl 
•	Average Delivery Time: ~7.86 days 
•	SLA Compliance Rate: 100% 
•	Shipment Count: 317 
 
7.	District with the Lowest Average Delivery Time: 
•	District: Mumbai 
•	Average Delivery Time: ~0.51 days 
•	SLA Compliance Rate: 100% 
•	Shipment Count: 40,572 
 
 
 ![image](https://github.com/user-attachments/assets/ff7690f9-61fa-41f9-bb59-d55dc2aa171e)

 ![image](https://github.com/user-attachments/assets/05d33e14-e8a7-474e-9a17-a32509425c51)

  
The chart below illustrates the shipment count trends over time, showing how order volumes fluctuate on a daily basis. This can help identify peak shipping days, seasonal trends, or irregularities in the shipping process 
  
 ![image](https://github.com/user-attachments/assets/ae743951-bafb-4d61-905b-24bfdc3ebb30)

Conclusion: 
The analysis highlights an efficient delivery system with an average time of ~2.49 days, though data inconsistencies (e.g., negative delivery times) need addressing. "Air" mode dominates in speed (~2.69 days) and volume, while surface modes are slower but likely cost-effective. High delays in areas like Khawzawl (~7.86 days) suggest logistical challenges, whereas regions like Mumbai (~0.51 days) demonstrate operational excellence. Shipment trends show seasonal fluctuations, highlighting the need for demand-based resource planning. 
Recommendations: 
•	Address data inconsistencies (e.g., negative delivery times and unrealistic SLA definitions) to enhance reporting accuracy and decision-making. 
•	Mode Optimization: Leverage the "Air" mode for high-priority shipments while improving surface-based modes for cost-effective deliveries. 
•	Regional Focus: Enhance delivery infrastructure and logistics in high-delay areas like Khawzawl. 
•	Study the efficient operations in Mumbai to replicate success in other regions. 
•	Demand Planning: Use shipment trends to predict peak periods and allocate resources effectively. 
 
 

 

