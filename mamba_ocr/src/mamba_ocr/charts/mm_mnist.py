per_token_train_loss = [
0.5068199050663367,
0.012645717320063001,
0.008052820826950247,
0.0011244495437810506,
0.005716131257399175,
0.005303218515132342,
0.00620103627883209,
0.000276865532222199,
3.6617692049150822e-06,
1.7925314090927854e-06,
1.006471385358151e-06,
5.815928575096068e-07,
3.394251447993197e-07,
1.9913802423721094e-07,
1.1729153822597522e-07,
6.929638108471581e-08,
4.105112912397965e-08,
2.4373956745783942e-08,
1.4442121064526882e-08,
8.654636246750526e-09,
]

in_context_accuracy = [
0.9958333333333333,
0.9979166666666667,
0.9958333333333333,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
]

shuffled_accuracy = [
0.996875,
0.996875,
0.9927083333333333,
0.9979166666666667,
1.0,
0.9989583333333333,
1.0,
0.9989583333333333,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
1.0,
0.9989583333333333,
1.0,
1.0,
]

import matplotlib.pyplot as plt

fig, ax = plt.subplots()

# Plot the data
ax.plot([epoch for epoch in range(1, 21)], in_context_accuracy, label="In-Context Accuracy")
ax.plot([epoch for epoch in range(1, 21)], shuffled_accuracy, label="Shuffled Accuracy")
ax.legend(loc='upper left')

# Customize the plot
ax.set_title("MedMamba Accuracy on Single Letters")
ax.set_xticks([epoch for epoch in range(1, 21)])
ax.set_xlabel("Epoch")
ax.set_ylabel("Accuracy")

# Show the plot
plt.show()