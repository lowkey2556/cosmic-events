"""ztf_clean writes the cleaned arrays and every step's counts in one pass: the expressions it relies on."""


def test_short_curves_keep_counts_but_no_arrays(spark):
    spark.sql("""SELECT 1L AS objectid, array(named_struct('hmjd', 1D, 'mag', 13F), named_struct('hmjd', 2D, 'mag', 14F))
                        AS z4""").createOrReplaceTempView("t_one_pass")
    r = spark.sql("""SELECT size(z4) AS nepochs, IF(size(z4) >= 2, transform(z4, x -> x.mag), NULL) AS mag,
                            IF(size(z4) >= 3, transform(z4, x -> x.mag), NULL) AS short,
                            IF(size(z4) >= 2, 1, 0) AS keep FROM t_one_pass""").first()
    assert (r["nepochs"], r["keep"], r["short"]) == (2, 1, None)
    assert list(r["mag"]) == [13.0, 14.0]
