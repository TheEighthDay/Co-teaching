import numpy as np
from scipy.stats import beta
import matplotlib.pyplot as plot

# 设置 plot 支持中文
from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)

# 定义一组alpha 跟 beta值
alpha_beta_values = [[10,1],[5,1], [1,1],[1,5],[1,10]]
linestyles = []

# 定义 x 值`
x = np.linspace(0, 1, 1002)[1:-1]
for alpha_beta_value in alpha_beta_values:
#   print(alpha_beta_value)
  dist = beta(alpha_beta_value[0], alpha_beta_value[1])
  dist_y = dist.pdf(x)
  # 添加图例
  # plot.legend('alpha=')
  # 创建 beta 曲线
  plot.plot(x, dist_y, label=r'$\alpha=%.1f,\ \beta=%.1f$' % (alpha_beta_value[0], alpha_beta_value[1]))

# 设置标题
plot.title('Beta')
# 设置 x,y 轴取值范围
plot.xlim(0,1)
plot.ylim(0, 10)
plot.legend()
plot.savefig("beta.png")
