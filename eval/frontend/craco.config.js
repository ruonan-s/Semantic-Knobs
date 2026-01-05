module.exports = {
  webpack: {
    configure: (webpackConfig) => {
      // Fix for the allowedHosts warning
      if (webpackConfig.devServer) {
        webpackConfig.devServer.allowedHosts = 'all';
      }
      
      // Suppress deprecation warnings
      webpackConfig.ignoreWarnings = [
        /DeprecationWarning/,
        /fs\.F_OK is deprecated/,
        /Critical dependency/,
        /Module not found/
      ];
      
      return webpackConfig;
    }
  },
  devServer: {
    allowedHosts: 'all',
    client: {
      overlay: {
        warnings: false,
        errors: true
      }
    },
    headers: {
      'Access-Control-Allow-Origin': '*'
    }
  },
  babel: {
    presets: [
      [
        '@babel/preset-env',
        {
          targets: {
            node: 'current'
          }
        }
      ]
    ]
  }
};


