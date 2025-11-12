
    x_values = [exp_df[param_x][i] for i in N_RUN]
    y_values = [exp_df[param_y][i] for i in N_RUN]
    
    #///// Create the Error Matrix:
    error = np.full((len(y_values), len(x_values)), np.nan)
    
    #///// Create a small supporting dict
    x_map = {param_x: i for i, param_x in enumerate(x_values)}
    y_map = {param_y: j for j, param_y in enumerate(y_values)}
    
    for exp in experiments:
        i = x_map[exp[param_x]]
        j = y_map[exp[param_y]]
        error[i, j] = exp["error"]