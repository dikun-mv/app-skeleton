const path = require('path');
const webpack = require('webpack');
const HtmlWebpackPlugin = require('html-webpack-plugin');

const root = `${__dirname}/..`;

module.exports = {
  mode: 'development',
  devtool: 'cheap-module-eval-source-map',
  entry: {
    app: path.resolve(root, 'src/index.js'),
    vendor: [
      '@tensorflow-models/posenet',
      '@tensorflow/tfjs',
      'axios',
      'stats.js'
    ],
  },
  plugins: [
    new webpack.EnvironmentPlugin(['NODE_ENV']),
    new HtmlWebpackPlugin({
      template: path.resolve(root, 'src/index.html'),
      favicon: path.resolve(root, 'src/favicon.ico'),
      title: 'App Skeleton'
    })
  ],
  optimization: {
    splitChunks: {
      cacheGroups: {
        vendor: {
          chunks: 'initial',
          test: 'vendor',
          name: 'vendor',
          enforce: true
        }
      }
    }
  },
  devServer: {
    host: 'localhost',
    port: 3000,
    open: true,
  }
};
