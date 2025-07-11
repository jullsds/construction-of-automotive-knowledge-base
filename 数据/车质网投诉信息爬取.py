import random
import time

from bs4 import BeautifulSoup
import pandas as pd
from fake_useragent import UserAgent
from selenium import webdriver

ua = UserAgent()
headers = {
    "User-Agent": ua.edge
}

# # 获取代理IP
# ips = []  # 建立数组，用于存放有效IP
# for i in range(1, 4):
#     ip_url = 'http://www.89ip.cn/index_%s.html'%i
#     # 请求IP的网站，得到源码
#     res = requests.get(url=ip_url, headers=headers).text
#     res_re = res.replace('\n', '').replace('\t', '').replace(' ', '')
#     # 使用正则表达匹配出IP地址及它的端口
#     re_c = re.compile('<tr><td>(.*?)</td><td>(.*?)</td><td>')
#     result = re.findall(re_c, res_re)
#     for i in range(len(result)):
#         ip = 'http://' + result[i][0] + ':' + result[i][1]
#         # 设置为字典格式
#         proxies = {"http": ip}
#         # 使用上面爬取的IP代理请求百度
#         html = requests.get('https://www.baidu.com/', proxies=proxies)
#         if html.status_code == 200:  # 状态码为200，说明请求成功
#             ips.append(proxies)  # 添加进数组中
# print("ip代理准备完成")


# # 典型问题
# data = pd.read_csv('车质网汽车投诉.csv')
#
#
# def split_str(row):
#     # 按照空格分割字符串
#     parts = row['典型问题'].split(' ')
#     # 返回分割后的列表
#     return [part for part in parts]
#
#
# data['典型问题'] = data.apply(split_str, axis=1)
#
# print(data['典型问题'])
# exit()


result = pd.DataFrame(columns=['投诉编号', '投诉品牌', '投诉车系', '投诉车型', '问题简述', '典型问题', '投诉时间', '投诉状态'])
browser = webdriver.Chrome()

for i in range(2600, 2700):
    # # eventlet.monkey_patch()
    # wait_seconds = random.uniform(0, 1)
    # time.sleep(wait_seconds)  # 随机等待
    url = 'https://www.12365auto.com/zlts/0-0-0-0-0-0_0-0-0-0-0-0-0-' + str(i) + '.shtml'
    # , proxies = ips[random.randint(0, len(ips) - 1)]

    browser.get(url)
    content = browser.page_source

    # content = requests.get(url, headers=headers).text
    soup = BeautifulSoup(content, 'html.parser')
    tr_list = soup.find('div', class_='tslb_b').find_all('tr')
    for tr in tr_list:
        tempMy = {}  # 先用个字典 准备存放每一个td里面的内容
        # 第一个tr没有td 因为它就是网页中的列索引名称 从第二个tr开始 里面才td 存放的是表格里的内容
        td_list = tr.find_all('td')
        if len(td_list) > 0:  # 如果有td 就去提取里面的内容

            # 典型问题提取
            td_a = td_list[5].find_all('a')
            cnt = 0
            ty_problem = ""
            for a in td_a:
                if cnt % 2 == 0:
                    ty_problem += a.text + '|'
                else:
                    ty_problem += a.text + ' '
                cnt += 1
            ty_problem = ty_problem.rstrip()
            # exit()

            tempMy['投诉编号'], tempMy['投诉品牌'], tempMy['投诉车系'], tempMy['投诉车型'], tempMy['问题简述'], tempMy['典型问题'], tempMy['投诉时间'], tempMy['投诉状态'] = \
                td_list[0].text, td_list[1].text, td_list[2].text, td_list[3].text, td_list[4].text.rstrip(), ty_problem, td_list[6].text, td_list[7].text
            temp = pd.DataFrame(tempMy, index=[0])  # 转换成Dataframe对象，便于合并
            result = pd.concat([result, temp], ignore_index=True)

browser.quit()
result.to_csv("车质网汽车投诉.csv", index=False, mode='a', header=False)
