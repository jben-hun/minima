---
layout: post
title:  "Steam server statistics"
categories: notebook
author:
- Jenei Bendegúz
excerpt: Source & Goldsource mod servers
---

<a href="https://colab.research.google.com/github/jben-hun/colab_notebooks/blob/master/SteamServerStat.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

This notebook uses the [Master Server Query Protocol](https://developer.valvesoftware.com/wiki/Master_Server_Query_Protocol) and [server queries](https://developer.valvesoftware.com/wiki/Server_queries#Protocol) to get a server list and statistics of multiplayer [Source](https://developer.valvesoftware.com/wiki/Source) and [Goldsource](https://developer.valvesoftware.com/wiki/Goldsource) engine mods.
*   This is for mods without an [AppID](https://developer.valvesoftware.com/wiki/Steam_Application_IDs) and own [Steam store page](https://developer.valvesoftware.com/wiki/Steam), for those official mods there are <https://store.steampowered.com/stats/> and <https://steamcharts.com/>
*   Python Steam API package documentation: <https://steam.readthedocs.io/en/stable/api/steam.game_servers.html>
*   Cool standalone server browser: <https://github.com/PredatH0r/SteamServerBrowser>


```python
# interactively display tables
%load_ext google.colab.data_table

%pip -q install steam
```


```python
import numpy as np
import pandas as pd
from steam import game_servers as gs
from collections import defaultdict


def get_server_list(*app_ids, has_players=True, limit=500):
    servers = []

    for id in app_ids:
        query_string = ""
        query_string += fr"\appid\{id}"
        query_string += fr"\empty\1" if has_players else ""
        sub = list(gs.query_master(query_string, max_servers=limit))
        servers += sub

        print(f"{len(sub)} servers from query: \"{query_string}\"")
        if len(sub) == limit:
            print(f"Warning, query response limit({limit}) reached, the "
                  f"result is possibly truncated, increase the limit")

    print(f"{len(servers)} servers total")

    return servers


def get_server_info(*servers, apps=None):
    infos = []
    for server in servers:
        try:
            info = gs.a2s_info(server)

            info["address"] = ":".join(map(str, server))

            if "app_id" in info:
                info["app"] = (apps[info["app_id"]] if apps is not None
                               else info["app_id"])
            else:
                info["app"] = "unknown"


            infos.append(info)

        except Exception as e:
            print(f"""{info["address"]}: {type(e).__name__}: {e}""")

    return (pd.DataFrame(infos)
            .set_index(["app", "folder"])
            .sort_values(by="players", ascending=False))


def display_player_count(info):
    display(
        info
        .groupby(info.index)
        .agg(players=pd.NamedAgg(column="players", aggfunc="sum"),
             populated_servers=pd.NamedAgg(column="players", aggfunc="count"),
             mean_players_per_server=pd.NamedAgg(column="players",
                                                 aggfunc="mean"),
             game_names=pd.NamedAgg(column="game", aggfunc=",".join))
        .sort_values(by="populated_servers", ascending=False)
    )


def display_server_info(info):
    c = ["address", "name", "app_id", "app", "game_id", "game", "folder",
         "map", "players", "max_players", "bots", "_type", "protocol",
         "server_type", "environment", "vac", "version", "keywords",
         "sourcetv_port", "sourcetv_name"]

    display(info.reset_index().loc[:, c])
```


```python
# Goldsource mods without a Steam page usually use Half-Life as the base game,
# Source mods use their respective Source SDK Bases
apps = {
    70: "Half-Life",
    215: "Source SDK Base 2006",
    218: "Source SDK Base 2007",
    243750: "Source SDK Base 2013 Multiplayer",
}
```


```python
servers = get_server_list(*apps.keys())
info = get_server_info(*servers, apps=apps)
```

    52 servers from query: "\appid\70\empty\1"
    9 servers from query: "\appid\215\empty\1"
    1 servers from query: "\appid\218\empty\1"
    9 servers from query: "\appid\243750\empty\1"
    71 servers total
    192.169.88.50:29035: timeout: timed out
    192.169.88.50:29035: RuntimeError: Invalid reponse header - b'D'
    18.156.63.218:27015: timeout: timed out
    78.46.191.68:27016: timeout: timed out
    78.46.191.68:27016: timeout: timed out
    78.46.191.68:27016: OSError: [Errno 113] No route to host
    78.46.191.68:27016: timeout: timed out
    78.46.191.68:27016: timeout: timed out
    78.46.191.68:27016: timeout: timed out
    78.46.191.68:27016: timeout: timed out
    78.46.191.68:27016: timeout: timed out
    78.46.191.68:27016: timeout: timed out
    78.46.191.68:27016: timeout: timed out
    78.46.191.68:27016: timeout: timed out
    78.46.191.68:27016: timeout: timed out
    185.86.78.191:27015: timeout: timed out
    185.86.78.191:27015: RuntimeError: Invalid reponse header - b'D'
    31.131.249.69:27018: RuntimeError: Invalid reponse header - b'D'
    89.223.32.156:27010: RuntimeError: Invalid reponse header - b'D'
    185.171.25.194:27015: timeout: timed out
    5.53.16.212:27015: timeout: timed out
    45.235.98.40:27042: timeout: timed out
    37.201.231.181:10894: timeout: timed out
    37.201.231.181:10894: timeout: timed out
    78.47.20.213:27019: timeout: timed out
    78.47.20.213:27019: timeout: timed out
    78.47.20.213:27019: timeout: timed out
    78.47.20.213:27019: timeout: timed out
    78.47.20.213:27019: timeout: timed out
    


```python
display_player_count(info)
display_server_info(info)
```


<div style="overflow: auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>players</th>
      <th>populated_servers</th>
      <th>mean_players_per_server</th>
      <th>game_names</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>(Half-Life, valve)</th>
      <td>143</td>
      <td>22</td>
      <td>6.500000</td>
      <td>Sev+|GG EDITION,D  A  W  N,H λ L F - L I F E,H...</td>
    </tr>
    <tr>
      <th>(Half-Life, ag)</th>
      <td>13</td>
      <td>6</td>
      <td>2.166667</td>
      <td>HLCCL,AG FFA,AG TDM,HLCCL,AGT,AG TDM</td>
    </tr>
    <tr>
      <th>(Source SDK Base 2006, hidden)</th>
      <td>4</td>
      <td>4</td>
      <td>1.000000</td>
      <td>Hidden : Source B4,Hidden : Source B4,Hidden :...</td>
    </tr>
    <tr>
      <th>(Source SDK Base 2013 Multiplayer, tf2classic)</th>
      <td>32</td>
      <td>4</td>
      <td>8.000000</td>
      <td>Team Fortress 2 Classic,Deathrun Toolkit | 0.4...</td>
    </tr>
    <tr>
      <th>(Source SDK Base 2006, gmod9)</th>
      <td>6</td>
      <td>2</td>
      <td>3.000000</td>
      <td>GMod 9.0.4,GMod 9.0.4</td>
    </tr>
    <tr>
      <th>(Half-Life, dpbredux)</th>
      <td>3</td>
      <td>1</td>
      <td>3.000000</td>
      <td>Digital Paintball</td>
    </tr>
    <tr>
      <th>(Half-Life, nnk)</th>
      <td>1</td>
      <td>1</td>
      <td>1.000000</td>
      <td>Naruto Naiteki Kensei R1</td>
    </tr>
    <tr>
      <th>(Source SDK Base 2006, zombie_master)</th>
      <td>2</td>
      <td>1</td>
      <td>2.000000</td>
      <td>Zombie Master 1.2.1</td>
    </tr>
    <tr>
      <th>(Source SDK Base 2007, gesource)</th>
      <td>3</td>
      <td>1</td>
      <td>3.000000</td>
      <td>Team Arsenal (MOD)</td>
    </tr>
  </tbody>
</table>
</div>



<div style="overflow: auto;">
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>address</th>
      <th>name</th>
      <th>app_id</th>
      <th>app</th>
      <th>game_id</th>
      <th>game</th>
      <th>folder</th>
      <th>map</th>
      <th>players</th>
      <th>max_players</th>
      <th>bots</th>
      <th>_type</th>
      <th>protocol</th>
      <th>server_type</th>
      <th>environment</th>
      <th>vac</th>
      <th>version</th>
      <th>keywords</th>
      <th>sourcetv_port</th>
      <th>sourcetv_name</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>54.39.130.18:27015</td>
      <td>VaultF4.com - Official [US] Standard</td>
      <td>243750</td>
      <td>Source SDK Base 2013 Multiplayer</td>
      <td>243750.0</td>
      <td>Team Fortress 2 Classic</td>
      <td>tf2classic</td>
      <td>cp_coldfront</td>
      <td>14</td>
      <td>30</td>
      <td>0</td>
      <td>source</td>
      <td>17</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>5394425</td>
      <td>HLstatsX:CE,alltalk,cp,increased_maxplayers,no...</td>
      <td>27020.0</td>
      <td>SourceTV Test</td>
    </tr>
    <tr>
      <th>1</th>
      <td>23.124.59.250:27015</td>
      <td>Gamers-Global.com|HLSTATS|SEV+|FASTDL|CUSTOM|1...</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Sev+|GG EDITION</td>
      <td>valve</td>
      <td>7th_path</td>
      <td>14</td>
      <td>22</td>
      <td>4</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.2</td>
      <td></td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>84.22.153.100:27018</td>
      <td>Dawn HL Crossfire Only</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>D  A  W  N</td>
      <td>valve</td>
      <td>crossfire</td>
      <td>14</td>
      <td>20</td>
      <td>3</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>188.213.212.5:27015</td>
      <td>|-&gt; -==[24/7]==- HL.Pyro-Zone.com &lt;-|</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>H λ L F - L I F E</td>
      <td>valve</td>
      <td>hl_assault</td>
      <td>13</td>
      <td>32</td>
      <td>2</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>62.140.250.10:27015</td>
      <td>! !--Good_Half-Life_Server--! !</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Half-Life</td>
      <td>valve</td>
      <td>crossfire</td>
      <td>13</td>
      <td>32</td>
      <td>1</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>5</th>
      <td>78.152.169.100:27016</td>
      <td>[VICTORY.KM.UA] Half-Life DM FFA</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>NaN</td>
      <td>Half-Life</td>
      <td>valve</td>
      <td>closefire</td>
      <td>13</td>
      <td>32</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>0</td>
      <td>1.1.2.7/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>192.223.30.68:27015</td>
      <td>! !--Best_Half-Life_Server--! !</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>H λ L F - L I F E</td>
      <td>valve</td>
      <td>crossfire</td>
      <td>10</td>
      <td>10</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>7</th>
      <td>84.22.153.100:27015</td>
      <td>Dawn Half-Life</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>D  A  W  N</td>
      <td>valve</td>
      <td>dead-dust2</td>
      <td>9</td>
      <td>20</td>
      <td>2</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>78.47.20.213:27019</td>
      <td>⬛█ Death Ventures | Combat / Jailbreak / Death...</td>
      <td>243750</td>
      <td>Source SDK Base 2013 Multiplayer</td>
      <td>243750.0</td>
      <td>Deathrun Toolkit | 0.4.1</td>
      <td>tf2classic</td>
      <td>dr_bank_v12a</td>
      <td>9</td>
      <td>24</td>
      <td>0</td>
      <td>source</td>
      <td>17</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>5394425</td>
      <td>deathrun,deathventures,jailbreak,teamfortress2...</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>9</th>
      <td>192.169.88.50:29035</td>
      <td>*L.T.K*Clan Server 1</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Severian's Mod</td>
      <td>valve</td>
      <td>dutch_wonderfire</td>
      <td>7</td>
      <td>26</td>
      <td>3</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>10</th>
      <td>51.195.91.179:27016</td>
      <td>VaultF4.com - Official [EU] Randomizer</td>
      <td>243750</td>
      <td>Source SDK Base 2013 Multiplayer</td>
      <td>243750.0</td>
      <td>Team Fortress 2 Classic</td>
      <td>tf2classic</td>
      <td>plr_hightower</td>
      <td>6</td>
      <td>24</td>
      <td>0</td>
      <td>source</td>
      <td>17</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>5394425</td>
      <td>HLstatsX:CE</td>
      <td>27021.0</td>
      <td>SourceTV Test</td>
    </tr>
    <tr>
      <th>11</th>
      <td>185.86.78.191:27015</td>
      <td>-=-=- hldm-ARENA.cf:27015 crossfire -=-=-</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Half-Life</td>
      <td>valve</td>
      <td>crossfire</td>
      <td>6</td>
      <td>24</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>12</th>
      <td>85.14.226.179:27015</td>
      <td>Xen-Raiders AG 2016 by ngz-server.de</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>HLCCL</td>
      <td>ag</td>
      <td>crossfire</td>
      <td>5</td>
      <td>18</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>13</th>
      <td>46.0.203.113:27015</td>
      <td>K_P_A_C_A_B_A  C_E_P_B_E_P</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>[ Music Server ]</td>
      <td>valve</td>
      <td>dk_katklub</td>
      <td>5</td>
      <td>18</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>0</td>
      <td>1.6.3.7/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>145.239.108.67:27015</td>
      <td>*L.T.K*Clan Server 3</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Severian's Mod</td>
      <td>valve</td>
      <td>tig_nite</td>
      <td>5</td>
      <td>26</td>
      <td>3</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>15</th>
      <td>139.162.49.132:27015</td>
      <td>! !--Asia_Half-Life_Server--! !</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>NaN</td>
      <td>H A L F - L I F E</td>
      <td>valve</td>
      <td>the_yard2</td>
      <td>5</td>
      <td>12</td>
      <td>2</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.7/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>16</th>
      <td>31.131.249.69:27018</td>
      <td>GunGame 2.2 - HLDM.ORG</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>NaN</td>
      <td>Half-Life</td>
      <td>valve</td>
      <td>ps2waypoint</td>
      <td>5</td>
      <td>32</td>
      <td>3</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.7/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>17</th>
      <td>5.53.16.212:27015</td>
      <td>[hlserv] DM + weaponmod</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>NaN</td>
      <td>Severian's Mod+</td>
      <td>valve</td>
      <td>crossfire</td>
      <td>5</td>
      <td>16</td>
      <td>1</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.7/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>18</th>
      <td>192.145.47.121:27017</td>
      <td>nafrayus server</td>
      <td>215</td>
      <td>Source SDK Base 2006</td>
      <td>NaN</td>
      <td>GMod 9.0.4</td>
      <td>gmod9</td>
      <td>gm_construct</td>
      <td>4</td>
      <td>16</td>
      <td>0</td>
      <td>source</td>
      <td>7</td>
      <td>d</td>
      <td>w</td>
      <td>0</td>
      <td>1.0.1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>19</th>
      <td>5.53.16.212:27017</td>
      <td>[hlserv] Crossfire 24\7</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>NaN</td>
      <td>Severian's Mod+</td>
      <td>valve</td>
      <td>crossfire</td>
      <td>4</td>
      <td>24</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.7/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>20</th>
      <td>89.223.32.156:27010</td>
      <td>AIMaster HL DM 1 | Newbies ONLY</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>NaN</td>
      <td>Half-Life</td>
      <td>valve</td>
      <td>crossfire</td>
      <td>4</td>
      <td>20</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.7/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>21</th>
      <td>94.23.247.37:27040</td>
      <td>HL2GO.COM - HLDM [EU]</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>NaN</td>
      <td>Half-Life</td>
      <td>valve</td>
      <td>dust2x2</td>
      <td>3</td>
      <td>32</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.7/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>22</th>
      <td>104.238.180.59:27065</td>
      <td>Official DPB Redux Field | USA | West</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Digital Paintball</td>
      <td>dpbredux</td>
      <td>NXL_TexasOpen_2018</td>
      <td>3</td>
      <td>20</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>23</th>
      <td>89.216.18.137:27025</td>
      <td>CobraNetwork: AGmod | FFA | Crossfire</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>AG FFA</td>
      <td>ag</td>
      <td>crossfire</td>
      <td>3</td>
      <td>10</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.2</td>
      <td></td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>24</th>
      <td>85.117.240.15:27025</td>
      <td>  ►  BlackWonder US TF2C | Hightower | No Car...</td>
      <td>243750</td>
      <td>Source SDK Base 2013 Multiplayer</td>
      <td>243750.0</td>
      <td>  ►  Hightower  ◄</td>
      <td>tf2classic</td>
      <td>plr_hightower</td>
      <td>3</td>
      <td>64</td>
      <td>0</td>
      <td>source</td>
      <td>17</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>5394425</td>
      <td>increased_maxplayers,_registered,afk,alltalk,b...</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>25</th>
      <td>74.91.114.38:27015</td>
      <td>Noob's Wonderland</td>
      <td>218</td>
      <td>Source SDK Base 2007</td>
      <td>218.0</td>
      <td>Team Arsenal (MOD)</td>
      <td>gesource</td>
      <td>ge_facility_classic</td>
      <td>3</td>
      <td>22</td>
      <td>0</td>
      <td>source</td>
      <td>17</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.0.1.0</td>
      <td>HLstatsX:CE,alltalk,autoteamplay,increased_max...</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>26</th>
      <td>45.155.125.20:27025</td>
      <td>SADECE PIMPLER</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Half-Life</td>
      <td>valve</td>
      <td>crossfire</td>
      <td>2</td>
      <td>16</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td></td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>27</th>
      <td>172.107.174.142:27015</td>
      <td>-=OAK=- Dallas, Texas</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Severian's Mod</td>
      <td>valve</td>
      <td>Boot2</td>
      <td>2</td>
      <td>16</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>28</th>
      <td>78.46.191.68:27016</td>
      <td>AGHLDM DE #2 | 1000 FPS | FastDL</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>AG TDM</td>
      <td>ag</td>
      <td>stalkx</td>
      <td>2</td>
      <td>32</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>29</th>
      <td>45.235.98.40:27042</td>
      <td>||LegenDary Team :-)</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>www.4evergaming.com.ar</td>
      <td>valve</td>
      <td>stalkx</td>
      <td>2</td>
      <td>12</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>30</th>
      <td>37.201.231.181:10894</td>
      <td>LEMCraft der pro</td>
      <td>215</td>
      <td>Source SDK Base 2006</td>
      <td>NaN</td>
      <td>GMod 9.0.4</td>
      <td>gmod9</td>
      <td>gm_construct</td>
      <td>2</td>
      <td>16</td>
      <td>0</td>
      <td>source</td>
      <td>7</td>
      <td>l</td>
      <td>w</td>
      <td>1</td>
      <td>1.0.1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>31</th>
      <td>149.28.44.79:27017</td>
      <td>-DTR- Almost Vanilla ZM Server (+RTD,RTV)</td>
      <td>215</td>
      <td>Source SDK Base 2006</td>
      <td>NaN</td>
      <td>Zombie Master 1.2.1</td>
      <td>zombie_master</td>
      <td>zm_zombiefacility_v2</td>
      <td>2</td>
      <td>16</td>
      <td>1</td>
      <td>source</td>
      <td>7</td>
      <td>d</td>
      <td>w</td>
      <td>0</td>
      <td>1.0.1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>32</th>
      <td>18.156.63.218:27015</td>
      <td>NPN Half-Life Server</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Half-Life</td>
      <td>valve</td>
      <td>crossfire</td>
      <td>1</td>
      <td>20</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.2</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>33</th>
      <td>185.107.96.114:27015</td>
      <td>SiDiouS Phys Server FF-Off PS-Off</td>
      <td>215</td>
      <td>Source SDK Base 2006</td>
      <td>215.0</td>
      <td>Hidden : Source B4</td>
      <td>hidden</td>
      <td>hdn_barrel_warz_v1</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>source</td>
      <td>7</td>
      <td>d</td>
      <td>w</td>
      <td>0</td>
      <td>1.0.1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>34</th>
      <td>74.91.124.33:27015</td>
      <td>The Living Dead - Hidden Source Server pshove/...</td>
      <td>215</td>
      <td>Source SDK Base 2006</td>
      <td>215.0</td>
      <td>Hidden : Source B4</td>
      <td>hidden</td>
      <td>hdn_demise</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>source</td>
      <td>7</td>
      <td>d</td>
      <td>w</td>
      <td>0</td>
      <td>1.0.1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>35</th>
      <td>162.248.93.98:27015</td>
      <td>Stabbiη Cabiη ♢ Ranked ♢ PS:Shove|FF:Reflect</td>
      <td>215</td>
      <td>Source SDK Base 2006</td>
      <td>215.0</td>
      <td>Hidden : Source B4</td>
      <td>hidden</td>
      <td>hdn_physics_arena_final</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>source</td>
      <td>7</td>
      <td>d</td>
      <td>w</td>
      <td>0</td>
      <td>1.0.1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>36</th>
      <td>173.24.229.150:27015</td>
      <td>Playground~Genuine~Gaming~HL~Linux</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Half-Life</td>
      <td>valve</td>
      <td>atticrats_ro</td>
      <td>1</td>
      <td>15</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>0</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>37</th>
      <td>74.91.113.158:27015</td>
      <td>OL'Farts Gaming www.olfartsgaming.com</td>
      <td>215</td>
      <td>Source SDK Base 2006</td>
      <td>215.0</td>
      <td>Hidden : Source B4</td>
      <td>hidden</td>
      <td>hdn_docks</td>
      <td>1</td>
      <td>9</td>
      <td>1</td>
      <td>source</td>
      <td>7</td>
      <td>d</td>
      <td>w</td>
      <td>0</td>
      <td>1.0.1.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>38</th>
      <td>95.216.140.175:27015</td>
      <td>{Helsinki} DM Server by [Z] Chuck Moris</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>Naruto Naiteki Kensei R1</td>
      <td>nnk</td>
      <td>nnk_preliminary</td>
      <td>1</td>
      <td>25</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>w</td>
      <td>1</td>
      <td>1.1.2.7/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>39</th>
      <td>112.213.38.144:27020</td>
      <td>AG Australia [1000fps]</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>HLCCL</td>
      <td>ag</td>
      <td>no_remorse</td>
      <td>1</td>
      <td>10</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td></td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>40</th>
      <td>185.171.25.194:27015</td>
      <td>[AGT] AGTurkiye Turnuva Sunucusu* |TR| #2</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>AGT</td>
      <td>ag</td>
      <td>datacore</td>
      <td>1</td>
      <td>32</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>d</td>
      <td>l</td>
      <td>1</td>
      <td>1.1.2.2/Stdio</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>41</th>
      <td>109.184.60.233:27015</td>
      <td>www.planethalflife.com/agmod</td>
      <td>70</td>
      <td>Half-Life</td>
      <td>70.0</td>
      <td>AG TDM</td>
      <td>ag</td>
      <td>ag_crossfire</td>
      <td>1</td>
      <td>32</td>
      <td>0</td>
      <td>source</td>
      <td>48</td>
      <td>l</td>
      <td>w</td>
      <td>0</td>
      <td>1.1.2.2/Stdio</td>
      <td></td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>
