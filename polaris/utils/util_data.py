import random
import warnings
import numpy as np
from scipy.sparse import coo_matrix
from torch.utils.data import Dataset
from scipy.sparse import SparseEfficiencyWarning
warnings.filterwarnings("ignore", category=SparseEfficiencyWarning)

def getLocal(mat, i, jj, w, N):
    if i >= 0 and jj >= 0 and i+w <= N and jj+w <= N:
        mat = mat[i:i+w,jj:jj+w].toarray()
        # print(f"global: {mat.shape}")
        return mat[None,...]
    # pad_width = ((up, down), (left, right))
    slice_pos = [[i, i+w], [jj, jj+w]]
    pad_width = [[0, 0], [0, 0]]
    if i < 0:
        pad_width[0][0] = -i
        slice_pos[0][0] = 0
    if jj < 0:
        pad_width[1][0] = -jj
        slice_pos[1][0] = 0
    if i+w > N:
        pad_width[0][1] = i+w-N
        slice_pos[0][1] = N
    if jj+w > N:
        pad_width[1][1] = jj+w-N
        slice_pos[1][1] = N
    _mat = mat[slice_pos[0][0]:slice_pos[0][1],slice_pos[1][0]:slice_pos[1][1]].toarray()
    padded_mat = np.pad(_mat, pad_width, mode='constant', constant_values=0)
    # print(f"global: {padded_mat.shape}",slice_pos, pad_width)
    return padded_mat[None,...]

def upperCoo2symm(row,col,data,N=None):
    # print(np.max(row),np.max(col),N)
    if N:
        shape=(N,N)
    else:
        shape=(row.max() + 1,col.max() + 1)

    sparse_matrix = coo_matrix((data, (row, col)), shape=shape)
    symm = sparse_matrix + sparse_matrix.T
    diagVal = symm.diagonal(0)/2
    symm = symm.tocsr()
    symm.setdiag(diagVal)
    return symm

def shuffleIFWithCount(df):
    shuffled_df = df.copy()
    shuffled_df[['oe', 'balanced']] = df[['oe', 'balanced']].sample(frac=1).reset_index(drop=True)
    return shuffled_df

def shuffleIF(df):
    if len(df)<10:
        df = shuffleIFWithCount(df)
        return df
    min=np.min(df['bin1_id'])
    max=np.max(df['bin1_id'])
    distance = df['distance'].iloc[0]
    bin1_id = np.random.randint(min, high=max, size=int(len(df)*1.5))
    bin2_id = bin1_id + distance
    pair_id = set(zip(bin1_id,bin2_id))
    if len(pair_id)<len(df)-50:
        bin1_id = np.random.randint(min, high=max, size=len(df))
        bin2_id = bin1_id + distance
        extra_pair_id = set(zip(bin1_id,bin2_id))
        pair_id.update(extra_pair_id)
    if len(pair_id)<len(df):
        df = df.sample(len(pair_id))
    pair_id = list(pair_id)
    random.shuffle(pair_id)
    pair_id=np.asarray(pair_id[:len(df)])
    df['bin1_id']=pair_id[:,0]
    df['bin2_id'] = pair_id[:,1]
    return df

class centerPredCoolDataset(Dataset):
    def __init__(self, coolfile, cchrom, step=224, w=224, max_distance_bin=600, decoy=False, restrictDecoy=False, s=0.9, raw=False):
        '''
        Args:
            step (int): the step of slide window moved and also the center crop size to predict 
        '''
        
        self.s=s
        oeMat, decoyOeMat, N = self._processCoolFile(coolfile, cchrom, decoy=decoy, restrictDecoy=restrictDecoy, raw=raw)
        self.data, self.i, self.j = self._prepare_data(oeMat, N, step, w, max_distance_bin, decoyOeMat)
        del oeMat, decoyOeMat

    def _prepare_data(self, oeMat, N, step, w, max_distance_bin, decoyOeMat=None):
        center_crop_size = step
        start_point = -(w - center_crop_size) // 2
        data, i_list, j_list = [], [], []
        joffset = np.repeat(np.linspace(0, w, w, endpoint=False, dtype=int)[np.newaxis, :], w, axis=0)
        ioffset = np.repeat(np.linspace(0, w, w, endpoint=False, dtype=int)[:, np.newaxis], w, axis=1)
        
        for i in range(start_point, N - w - start_point, step):
            _data, _i_list, _j_list = self._process_window(oeMat, i, step, w, N, joffset, ioffset, max_distance_bin, decoyOeMat)
            data.extend(_data)
            i_list.extend(_i_list)
            j_list.extend(_j_list)
        
        return data, i_list, j_list
    
    def _process_window(self, oeMat, i, step, w, N, joffset, ioffset, max_distance_bin, decoyOeMat=None):
        data, i_list, j_list = [], [], []
        for j in range(0, max_distance_bin, step):
            jj = j + i
            # if jj + w <= N and i + w <= N:
            _oeMat = getLocal(oeMat, i, jj, w, N)
            if np.sum(_oeMat == 0) <= (w*w*self.s):
                if decoyOeMat is not None:
                    _decoyOeMat = getLocal(decoyOeMat, i, jj, w, N)
                    data.append(np.vstack((_oeMat, _decoyOeMat)))
                else:
                    data.append(_oeMat)
                    
                i_list.append(i + ioffset)
                j_list.append(jj + joffset)
        return data, i_list, j_list
    
    def _processCoolFile(self, coolfile, cchrom, decoy=False, restrictDecoy=False, raw=False):
        extent = coolfile.extent(cchrom)
        N = extent[1] - extent[0]
        if raw:
            ccdata = coolfile.matrix(balance=False, sparse=True, as_pixels=True).fetch(cchrom)
            v='count'
        else:
            ccdata = coolfile.matrix(balance=True, sparse=True, as_pixels=True).fetch(cchrom)
            v='balanced'
        ccdata['bin1_id'] -= extent[0]
        ccdata['bin2_id'] -= extent[0]
            
        ccdata['distance'] = ccdata['bin2_id'] - ccdata['bin1_id']
        d_means = ccdata.groupby('distance')[v].transform('mean')
        ccdata[v] = ccdata[v].fillna(0)
        
        ccdata['oe'] = ccdata[v] / d_means
        ccdata['oe'] = ccdata['oe'].fillna(0)
        ccdata['oe'] = ccdata['oe'] / ccdata['oe'].max()
        oeMat = upperCoo2symm(ccdata['bin1_id'].ravel(), ccdata['bin2_id'].ravel(), ccdata['oe'].ravel(), N)
        
        decoyMat = None
        if decoy:
            decoydata = ccdata.copy(deep=True)
            np.random.seed(0)
            if restrictDecoy:
                decoydata = decoydata.groupby('distance').apply(shuffleIF)
            else:
                decoydata = decoydata.groupby('distance').apply(shuffleIFWithCount)
                
            decoyMat = upperCoo2symm(decoydata['bin1_id'].ravel(), decoydata['bin2_id'].ravel(), decoydata['oe'].ravel(), N)
            
        return oeMat, decoyMat, N
            
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.i[idx], self.j[idx], self.data[idx]