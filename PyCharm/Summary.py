import numpy as np
from scipy.integrate import cumtrapz
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
import scipy.stats as stats
import matplotlib.patches as mpatches
from scipy.stats import norm
from math import ceil

###------------------------------------ Create theoretical population -----------------------------------------------###
lowest_height_whole_pop= 55
highest_height_whole_pop = 85
mean_height_whole_pop = 70
std_height_whole_pop = 3
num_bars_whole_pop = 100
n_points_x_axis = 1000

def normal_distribution(x, mu, sigma):
    pdf_values = (1 / (sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)
    return pdf_values

def create_whole_population(f_normal_distribution, lowest_height, highest_height, mean_height, std_height, n_points):
    x_values = np.linspace(lowest_height, highest_height, n_points)
    pdf_curve = [f_normal_distribution(x, mean_height, std_height) for x in x_values]
    return pd.DataFrame({'Height': x_values, 'Density': pdf_curve})

def plot_whole_population(df, num_bins, lowest_height, highest_height):
    fig, ax = plt.subplots(figsize=(4, 4))
    bin_width = (highest_height - lowest_height) / num_bins
    mid_x_values = np.linspace(lowest_height, highest_height, num_bins) + bin_width / 2
    pdf_values = [df['Density'].iloc[np.argmin(np.abs(df['Height'] - x))] for x in mid_x_values]
    ax.plot(df['Height'], df['Density'], label='PDF', color='red', linestyle='dashed', alpha=0.5, linewidth=1.5)
    ax.bar(mid_x_values, pdf_values, width=bin_width, alpha=0.5, edgecolor='black', linewidth=0.5, label='Histogram')
    ax.set_title('Whole Population of Human Heights')
    ax.set_xlabel('Height (inches)')
    ax.set_ylabel('Density')
    ax.legend()
    plt.tight_layout()
    plt.show()

# Create whole population
df_whole_pop = create_whole_population(normal_distribution, lowest_height_whole_pop, highest_height_whole_pop,
                                       mean_height_whole_pop, std_height_whole_pop, n_points_x_axis)

# Plot whole population with histogram bars
plot_whole_population(df_whole_pop, num_bars_whole_pop, lowest_height_whole_pop, highest_height_whole_pop)

###------------------------------------ Create mini population 01 ---------------------------------------------------###
lowest_height_mini_pop_01 = 55
mean_height_mini_pop_01 = 70
highest_height_mini_pop_01 = 85
std_height_mini_pop_01 = 3
num_bars_mini_pop_01 = 100
# approx_num_bars_mini_pop_01=9
# approx_total_count_mini_pop_01=30
approx_num_bars_mini_pop_01 = 5
approx_total_count_mini_pop_01 = 30


def create_mini_population(f_normal_distribution, start, end, n_bins, total_count, mean_height, std_height):
    bin_edges = np.linspace(start, end, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0
    bin_centers = np.round(bin_centers, 2)  # Round bin_centers to two decimal places
    bin_width = (end - start) / n_bins
    densities = np.array(
        [np.mean(f_normal_distribution(np.linspace(edge, bin_edges[i + 1], 100), mean_height, std_height)) for i, edge
         in enumerate(bin_edges[:-1])])
    float_counts = densities * total_count / sum(densities)
    int_counts = np.round(float_counts).astype(int)
    non_zero_indices = int_counts != 0
    bin_centers = bin_centers[non_zero_indices]
    int_counts = int_counts[non_zero_indices]
    probabilities = int_counts / sum(int_counts)  # Calculate probabilities
    return pd.DataFrame({'Height': bin_centers, 'Count': int_counts, 'Probability': probabilities})


def plot_mini_population(df, ax, label):
    centers = df['Height']
    counts = df['Count']
    bin_width = centers.iloc[1] - centers.iloc[0]
    ax.bar(centers, counts, width=bin_width, alpha=0.5, edgecolor='black', linewidth=0.5, label=label)


def calculate_weighted_mean(df_col_value, df_col_count):
    weighted_mean = (df_col_value * df_col_count).sum() / df_col_count.sum()
    return weighted_mean


def calculate_weighted_std(df_col_value, df_col_count):
    weighted_mean = (df_col_value * df_col_count).sum() / df_col_count.sum()
    squared_deviations = ((df_col_value - weighted_mean) ** 2) * df_col_count
    weighted_variance = squared_deviations.sum() / df_col_count.sum()
    weighted_std = np.sqrt(weighted_variance)
    return weighted_std


# Create mini (discretized) population
df_mini_pop_01 = create_mini_population(normal_distribution, lowest_height_mini_pop_01, highest_height_mini_pop_01,
                                        approx_num_bars_mini_pop_01, approx_total_count_mini_pop_01,
                                        mean_height_mini_pop_01, std_height_mini_pop_01)

# Display mini population
print(df_mini_pop_01)

# Plot mini population
fig, ax = plt.subplots(figsize=(4, 4))
plot_mini_population(df_mini_pop_01, ax, "Population 1")
ax.set_title('Mini Population of Human Heights')
ax.set_xlabel('Height (inches)')
ax.set_ylabel('Count')
plt.tight_layout()
plt.show()

# Summary
mean_mini_pop_01 = calculate_weighted_mean(df_mini_pop_01['Height'], df_mini_pop_01['Count'])
std_mini_pop_01 = calculate_weighted_std(df_mini_pop_01['Height'], df_mini_pop_01['Count'])

print(f"Unique elements: {df_mini_pop_01['Height'].unique()}")
print(f"Total count of elements: {df_mini_pop_01['Count'].sum()}")
print(f"Mean: {mean_mini_pop_01}")
print(f"Std: {std_mini_pop_01:.2f}")


###------------------------------------ Create mini population 02 ---------------------------------------------------###
lowest_height_mini_pop_02 = 75
mean_height_mini_pop_02 = 90
highest_height_mini_pop_02 = 105
std_height_mini_pop_02 = 3
num_bars_mini_pop_02 = 100
approx_num_bars_mini_pop_02 = 9
approx_total_count_mini_pop_02 = 30


def plot_two_mini_pops(df1, df2):
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot the first mini population
    plot_mini_population(df1, ax, "Population 1")

    # Plot the second mini population
    plot_mini_population(df2, ax, "Population 2")

    ax.set_title('Mini Populations of Human Heights (Not realistic)')
    ax.set_xlabel('Height (inches)')
    ax.set_ylabel('Count')
    ax.legend()
    plt.tight_layout()
    plt.show()


# Create mini (discretized) population
df_mini_pop_02 = create_mini_population(normal_distribution, lowest_height_mini_pop_02, highest_height_mini_pop_02,
                                        approx_num_bars_mini_pop_02, approx_total_count_mini_pop_02,
                                        mean_height_mini_pop_02, std_height_mini_pop_02)

# Summary
# Mini pop 01
mean_mini_pop_01 = calculate_weighted_mean(df_mini_pop_01['Height'], df_mini_pop_01['Count'])
std_mini_pop_01 = calculate_weighted_std(df_mini_pop_01['Height'], df_mini_pop_01['Count'])

print('Mini pop 01')
print(df_mini_pop_01)
# print(f"Unique elements: {df_mini_pop_01['Height'].unique()}")
print(f"Total count of elements: {df_mini_pop_01['Count'].sum()}")
print(f"Mean: {mean_mini_pop_01}")
print(f"Std: {std_mini_pop_01:.2f}")

print('')

# Mini pop 02
mean_mini_pop_02 = calculate_weighted_mean(df_mini_pop_02['Height'], df_mini_pop_02['Count'])
std_mini_pop_02 = calculate_weighted_std(df_mini_pop_02['Height'], df_mini_pop_02['Count'])

print('Mini pop 02')
print(df_mini_pop_02)
# print(f"Unique elements: {df_mini_pop_02['Height'].unique()}")
print(f"Total count of elements: {df_mini_pop_02['Count'].sum()}")
print(f"Mean: {mean_mini_pop_02}")
print(f"Std: {std_mini_pop_02:.2f}")

# Plot mini population
plot_two_mini_pops(df_mini_pop_01, df_mini_pop_02)


###------------------------------------- SDSM of the mini population 01 ---------------------------------------------###
sample_size_mini_pop_01 = 18

def create_df_all_combinations_with_replacement(df, sample_size):
    all_combs = list(product(df.iloc[:, 0].unique(), repeat=sample_size))
    columns = [f"Score {i + 1}" for i in range(sample_size)] + ['Mean']
    df_combinations = pd.DataFrame(all_combs, columns=columns[:-1])
    df_combinations['Mean'] = df_combinations.mean(axis=1)
    first_index = "Combination(s) 1"
    other_indices = [f"{i + 1}" for i in range(1, len(all_combs))]
    df_combinations.index = [first_index] + other_indices
    return df_combinations


def calculate_SE_efficiently(df, sample_size):
    unique_scores = df['Height'].unique()
    all_combs = np.array(list(product(unique_scores, repeat=sample_size)), dtype=np.float64)
    means = np.mean(all_combs, axis=1)
    SE = np.std(means)
    return SE


def rectify_df_combinations(df_combinations):
    # Initialize the new Mean_rectified column with zeros
    df_combinations['Mean_rectified'] = 0.0

    # Calculate the new Mean_rectified values
    for idx, row in df_combinations.iterrows():
        current_mean = row['Mean']
        current_mean_ceiled = round(ceil(current_mean * 100) / 100, 2)

        is_within_0_01 = np.abs(
            df_combinations['Mean'].apply(lambda x: round(ceil(x * 100) / 100, 2)) - current_mean_ceiled) < 0.011
        is_greater = df_combinations['Mean'].apply(lambda x: round(ceil(x * 100) / 100, 2)) > current_mean_ceiled

        mean_adjusted = (is_within_0_01 & is_greater).any()

        df_combinations.at[idx, 'Mean_rectified'] = current_mean_ceiled + 0.01 if mean_adjusted else current_mean_ceiled

    return df_combinations


def generate_mean_summary_df(df_combinations):
    mean_summary = df_combinations.groupby('Mean').size().reset_index(name='Total_count')
    mean_summary['Probability'] = mean_summary['Total_count'] / mean_summary['Total_count'].sum()
    mean_summary['Probability'] = mean_summary['Probability'].round(4)
    return mean_summary


def plot_sdsm(df_combinations, column_name='Mean'):
    # Count the number of columns containing the word "score" in their names
    sample_size = sum(df_combinations.columns.str.contains('score', case=False))

    # Group by the mean and count the occurrences
    mean_counts = df_combinations.groupby(column_name).size().reset_index(name='Counts')

    # Plotting
    fig, ax = plt.subplots(figsize=(3, 2.5))

    ax.scatter(mean_counts[column_name], mean_counts['Counts'], alpha=0.7, s=50)
    ax.plot(mean_counts[column_name], mean_counts['Counts'], color='blue', linewidth=0.8)
    for _, row in mean_counts.iterrows():
        ax.plot([row[column_name], row[column_name]], [0, row['Counts']], linestyle='--', color='grey', linewidth=0.5)
    ax.set_title(f"Sampling distribution of sample means (n = {sample_size})")
    ax.set_xlabel(column_name)
    ax.set_ylabel('Count')

    if len(mean_counts[column_name]) > 30:
        ax.set_xticks(np.linspace(mean_counts[column_name].min(), mean_counts[column_name].max(), 10))
    else:
        ax.set_xticks(mean_counts[column_name].values)

    ax.set_ylim(0, )

    if len(mean_counts[column_name]) > 4:
        ax.tick_params(axis='x', rotation=90)

    plt.tight_layout()
    plt.show()


def compare_empirical_estimated_SE(SE, pop_std, sample_size):
    estimated_std = pop_std / np.sqrt(sample_size)
    print(f"mini pop std: {pop_std:.2f}")
    print(f"Empirical SE: {SE:.2f}")
    print(f"Estimated SE from mini pop std: {estimated_std:.2f}")


SE_mini_pop_01 = calculate_SE_efficiently(df_mini_pop_01, sample_size_mini_pop_01)
# # Create all possible score combinations and calculate their means of mini pop 01
# df_sdsm_mini_pop_01_temp = create_df_all_combinations_with_replacement(df_mini_pop_01, sample_size_mini_pop_01)

# # Rectify the rounding in df_combinations
# df_sdsm_mini_pop_01 = rectify_df_combinations(df_sdsm_mini_pop_01_temp)

# # Display all possible combinations of mini pop 01
# display(df_sdsm_mini_pop_01)

# # Plot the SDMS (Sampling distribution of the sample means) of mini pop 01
# plot_sdsm(df_sdsm_mini_pop_01, 'Mean_rectified')

# # Summary
# mean_summary_df_mini_pop_01 = generate_mean_summary_df(df_sdsm_mini_pop_01)

# mean_sdms_mini_pop_01 = calculate_weighted_mean(mean_summary_df_mini_pop_01['Mean'], mean_summary_df_mini_pop_01['Total_count']).round(4)
# std_sdms_mini_pop_01 = calculate_weighted_std(mean_summary_df_mini_pop_01['Mean'], mean_summary_df_mini_pop_01['Total_count']).round(4)

# print('SDSM mini pop 01')
# display(mean_summary_df_mini_pop_01)
# #print(f"Unique elements: {mean_summary_df_mini_pop_01['Mean'].unique()}")
# print(f"Total count of elements: {mean_summary_df_mini_pop_01['Total_count'].sum()}")
# print(f"Mean: {mean_sdms_mini_pop_01}")
# print(f"SE: {std_sdms_mini_pop_01:.2f}")
# print("")
#compare_empirical_estimated_SE(SE_mini_pop_01, std_mini_pop_01, sample_size_mini_pop_01)

print("test")