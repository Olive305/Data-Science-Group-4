"""
# Show the values (x and y) in a graph to look at their coherence
# Each value pair as a dot in the graph
if False:
    plt.scatter(X, y, alpha=0.5)
    plt.title('Zusatzbeitrag_diff vs Mitglieder_diff_next')
    plt.xlabel('Zusatzbeitrag_diff')
    plt.ylabel('Mitglieder_diff_next')
    plt.grid(True)
    plt.show()


# Print the first 5 rows of the dataframe
print(df.head())

# Create a table which shows the price increase over time
if True:
    plt.figure(figsize=(10, 6))
    for name, group in df.groupby('Krankenkasse'):
        plt.plot(group['Date'], group['Zusatzbeitrag_diff'], marker='o', label=name, alpha=0.7)

    plt.title('Price Increase Over Time')
    plt.xlabel('Time')
    plt.ylabel('Price Increase (Zusatzbeitrag_diff)')
    plt.legend(loc='best', fontsize='small', title='Krankenkasse')
    plt.grid(True)
    plt.show()

"""