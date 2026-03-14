const path = require('path');
const HtmlWebpackPlugin = require('html-webpack-plugin');
const MiniCssExtractPlugin = require('mini-css-extract-plugin');

module.exports = (env, argv) => {
  const isProduction = argv.mode === 'production';
  
  return {
    entry: './src/App.fs.js',
    output: {
      path: path.resolve(__dirname, 'dist'),
      filename: isProduction ? '[name].[contenthash].js' : '[name].js',
      clean: true,
    },
    devServer: {
      static: './dist',
      hot: true,
      port: 3000,
      historyApiFallback: true,
    },
    module: {
      rules: [
        {
          test: /\.fs$/,
          use: {
            loader: 'fable-loader',
            options: {
              babel: {
                presets: ['@babel/preset-react'],
              },
            },
          },
        },
        {
          test: /\.js$/,
          exclude: /node_modules/,
          use: {
            loader: 'babel-loader',
            options: {
              presets: ['@babel/preset-env', '@babel/preset-react'],
            },
          },
        },
        {
          test: /\.css$/,
          use: [
            isProduction ? MiniCssExtractPlugin.loader : 'style-loader',
            'css-loader',
          ],
        },
      ],
    },
    plugins: [
      new HtmlWebpackPlugin({
        template: './public/index.html',
        title: 'TARS Dynamic UI - Agent Generated',
      }),
      ...(isProduction ? [new MiniCssExtractPlugin()] : []),
    ],
    resolve: {
      extensions: ['.fs', '.js', '.json'],
    },
    devtool: isProduction ? 'source-map' : 'eval-source-map',
  };
};
