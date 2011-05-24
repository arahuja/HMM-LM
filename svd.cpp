public class SVD
{

private:
	int m, n;
	double** u;
	double** v;
	double* w;
	double eps;
	double tsh;

public:
	SVD(double** a_, int m_, int n_)
	{
		_m = m_;
		_n = n_;
		_eps = .00001;
		decompose();
		reorder();
		_tsh = .5 * sqrt(m+n+1)*w[0]*_eps;
	}

	void solve();

	int rank(double thresh);
	int nullity(double thresh);
	double** range(double thresh);
	double** nullspace(double thresh);

	inline double invCondition()
	{
		return (w[0] <= 0 || w[_n-1] <= 0) ? 0 : w[_n-1] / w[0];
	}


	void decompose();
	void reorder();
	double pythag(double a, double b);

};

int SVD::rank(double thresh = .001)
{
	int rank = 0;
	for (int i =0; i <n; i++)
	{
		if (w[i]) > _thresh) nr++;
	}
	return rank;
}

int SVD::nullity(double thresh = .001)
{
	int nullity = 0;
	for (int i =0; i < _n; i++)
	{
		if (w[i]) <= _thresh) nullity++;
	}
	return nullity;
}

double ** range(double thresh_ = -1)
{
	int nr = 0;
	double ** range = createMatrix(_m, rank());
	for (int j= 0; j< _n; j++)
	{
		if (w[i]) > _thresh)
		{
			for (int i = 0; i<_m; i++)
			{
				range[i][nr] = u[i][j]
			}
			nr++;
		}
	}

	return range;
}

double** SVD::nullspace()
{
	double** nullspace = createMatrix(_n, nullity());
	int nn = 0;
	for (int j = 0; j < _n; j++)
	{
		if (w[j] <= _thresh)
		{
			for (int i = 0; i < _n; i++)
			{
				nullspace[i][nn] = v[i][j];
			}
			nn++;
		}
	}

	return nullspace;
}

int SVD::solve(double *b_, double *x_, int len)
{
	if (_len != _m) return -1;
	double * temp  = new double[_n];
	for (int j = 0; j< _n; j++)
	{
		double s = 0;
		if (w[j] > _thresh)
		{
			for (int i = 0; i< _m; i++)
			{
				s += u[i][j] * b[i];
			}
			s /= w[j];
		}
		temp[j] = s;
	}
	for (int j = 0; j < _n; j++)
	{
		double s = 0;
		for (int jj =0 ; jj < _n; jj++)
		{

		}
		x[j] = s;
	}
}

