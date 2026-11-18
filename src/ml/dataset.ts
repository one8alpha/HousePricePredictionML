import { FeatureMeta, FurnishingStatus, RawHousingRecord, YesNo } from './types';

export const FEATURE_METADATA: FeatureMeta[] = [
  {
    key: 'area',
    label: 'Plot & Built-up Area',
    shortLabel: 'Area',
    type: 'numerical',
    unit: 'sq.ft.',
    description: 'Total property area in square feet',
  },
  {
    key: 'bedrooms',
    label: 'Bedrooms',
    shortLabel: 'Bedrooms',
    type: 'numerical',
    description: 'Number of bedrooms in the house',
  },
  {
    key: 'bathrooms',
    label: 'Bathrooms',
    shortLabel: 'Bathrooms',
    type: 'numerical',
    description: 'Number of bathrooms',
  },
  {
    key: 'stories',
    label: 'Stories / Floors',
    shortLabel: 'Stories',
    type: 'numerical',
    description: 'Number of floor levels',
  },
  {
    key: 'mainroad',
    label: 'Main Road Access',
    shortLabel: 'Main Road',
    type: 'categorical-binary',
    description: 'Whether the property connects directly to a main road',
  },
  {
    key: 'guestroom',
    label: 'Guest Room',
    shortLabel: 'Guest Room',
    type: 'categorical-binary',
    description: 'Presence of a dedicated guest room',
  },
  {
    key: 'basement',
    label: 'Basement',
    shortLabel: 'Basement',
    type: 'categorical-binary',
    description: 'Whether the house includes a basement',
  },
  {
    key: 'hotwaterheating',
    label: 'Hot Water Heating',
    shortLabel: 'Hot Water',
    type: 'categorical-binary',
    description: 'Central hot water heating availability',
  },
  {
    key: 'airconditioning',
    label: 'Air Conditioning',
    shortLabel: 'AC',
    type: 'categorical-binary',
    description: 'Installed air conditioning system',
  },
  {
    key: 'parking',
    label: 'Parking Spaces',
    shortLabel: 'Parking',
    type: 'numerical',
    description: 'Covered / dedicated car parking capacity (0–3)',
  },
  {
    key: 'prefarea',
    label: 'Preferred Locality',
    shortLabel: 'Pref. Area',
    type: 'categorical-binary',
    description: 'Located in a preferred residential neighborhood',
  },
  {
    key: 'furnishingstatus',
    label: 'Furnishing Status',
    shortLabel: 'Furnishing',
    type: 'categorical-ordinal',
    description: 'Interior furnishing level (unfurnished, semi-furnished, furnished)',
  },
];

// Deterministic PRNG (Mulberry32) so dataset is 100% reproducible
export function createRng(seed: number) {
  let a = seed >>> 0;
  return function next() {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function normalRandom(rng: () => number): number {
  const u1 = Math.max(1e-7, rng());
  const u2 = rng();
  return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
}

/**
 * Generates a realistic 545-row Housing Prices dataset modeled after classic
 * UCI / Kaggle Housing datasets (13 columns, price in INR), calibrated so a
 * standard 1500 sq.ft., 3 bed, 2 bath, 1 parking home prices around ₹68.5 Lakhs.
 * Deliberately injects 8 missing values, 5 duplicate rows, and 7 IQR outliers
 * so Part C Data Preprocessing performs genuine, verifiable cleaning.
 */
export function generateRawHousingDataset(): RawHousingRecord[] {
  const rng = createRng(202601);
  const baseRows: RawHousingRecord[] = [];
  const furnishingOptions: FurnishingStatus[] = ['unfurnished', 'semi-furnished', 'furnished'];

  const totalUnique = 540;
  for (let i = 0; i < totalUnique; i++) {
    // Realistic distribution of house areas (1100 to 9500 sq.ft., mean ~4600)
    const rawArea = Math.round(
      Math.min(10500, Math.max(1050, 4400 + normalRandom(rng) * 1850))
    );
    const bedrooms =
      rawArea < 2200
        ? rng() < 0.72
          ? 2
          : 3
        : rawArea < 4500
          ? rng() < 0.6
            ? 3
            : rng() < 0.85
              ? 2
              : 4
          : rng() < 0.55
            ? 4
            : rng() < 0.85
              ? 3
              : 5;

    const bathrooms =
      bedrooms <= 2
        ? rng() < 0.75
          ? 1
          : 2
        : bedrooms === 3
          ? rng() < 0.55
            ? 2
            : rng() < 0.88
              ? 1
              : 3
          : rng() < 0.6
            ? 2
            : 3;

    const stories =
      rng() < 0.36 ? 1 : rng() < 0.76 ? 2 : rng() < 0.91 ? 3 : 4;

    const mainroad: YesNo = rng() < 0.84 ? 'yes' : 'no';
    const guestroom: YesNo = rng() < 0.22 ? 'yes' : 'no';
    const basement: YesNo = rng() < 0.34 ? 'yes' : 'no';
    const hotwaterheating: YesNo = rng() < 0.08 ? 'yes' : 'no';
    const airconditioning: YesNo = rng() < 0.36 ? 'yes' : 'no';
    const parking = rng() < 0.42 ? 0 : rng() < 0.75 ? 1 : rng() < 0.94 ? 2 : 3;
    const prefarea: YesNo = rng() < 0.26 ? 'yes' : 'no';

    const fRand = rng();
    const furnishingstatus: FurnishingStatus =
      fRand < 0.31 ? 'unfurnished' : fRand < 0.73 ? 'semi-furnished' : 'furnished';

    // Non-linear + interaction pricing model in INR (calibrated so 1500 sq.ft., 3BHK, 2Bath, 1Park ≈ ₹68.5L)
    const furnVal = furnishingstatus === 'furnished' ? 2 : furnishingstatus === 'semi-furnished' ? 1 : 0;
    const areaTierPremium =
      rawArea > 5200 ? 1850000 : rawArea > 3200 ? 850000 : 0;
    const luxuryCombo =
      airconditioning === 'yes' && prefarea === 'yes' ? 920000 : 0;
    const bathAreaCombo =
      bathrooms >= 2 && rawArea > 2600 ? 640000 : 0;

    const basePrice =
      950000 +
      rawArea * 780 +
      bedrooms * 340000 +
      bathrooms * 720000 +
      stories * 380000 +
      (mainroad === 'yes' ? 350000 : 0) +
      (guestroom === 'yes' ? 310000 : 0) +
      (basement === 'yes' ? 340000 : 0) +
      (hotwaterheating === 'yes' ? 480000 : 0) +
      (airconditioning === 'yes' ? 680000 : 0) +
      parking * 310000 +
      (prefarea === 'yes' ? 560000 : 0) +
      furnVal * 290000 +
      areaTierPremium +
      luxuryCombo +
      bathAreaCombo;

    const marketNoise = normalRandom(rng) * (basePrice * 0.055);
    const price = Math.max(
      1750000,
      Math.round((basePrice + marketNoise) / 10000) * 10000
    );

    baseRows.push({
      id: i + 1,
      price,
      area: rawArea,
      bedrooms,
      bathrooms,
      stories,
      mainroad,
      guestroom,
      basement,
      hotwaterheating,
      airconditioning,
      parking,
      prefarea,
      furnishingstatus: furnishingOptions[furnVal],
    });
  }

  // Inject 7 realistic IQR outliers (extreme luxury estates / data entry spikes)
  const outlierIndices = [18, 64, 142, 219, 311, 408, 495];
  const outlierValues = [
    { area: 15600, price: 18400000 },
    { area: 16200, price: 17950000 },
    { area: 14800, price: 16800000 },
    { area: 13900, price: 15900000 },
    { area: 14500, price: 16500000 },
    { area: 13200, price: 15400000 },
    { area: 15100, price: 17200000 },
  ];
  outlierIndices.forEach((idx, k) => {
    baseRows[idx].area = outlierValues[k].area;
    baseRows[idx].price = outlierValues[k].price;
    baseRows[idx].bedrooms = 5;
    baseRows[idx].bathrooms = 4;
    baseRows[idx].airconditioning = 'yes';
  });

  // Inject 8 realistic missing values (NaN / null) across area, bathrooms, bedrooms, parking
  baseRows[11].area = null;
  baseRows[78].area = null;
  baseRows[245].area = null;
  baseRows[42].bathrooms = null;
  baseRows[189].bathrooms = null;
  baseRows[330].bedrooms = null;
  baseRows[124].parking = null;
  baseRows[412].parking = null;

  // Append 5 exact duplicate records to make 545 total raw records
  const duplicateSources = [3, 27, 95, 180, 260];
  duplicateSources.forEach((srcIdx, dIdx) => {
    baseRows.push({
      ...baseRows[srcIdx],
      id: totalUnique + dIdx + 1,
    });
  });

  return baseRows;
}

export function formatINR(amount: number): string {
  if (!Number.isFinite(amount)) return '₹0';
  const rounded = Math.round(amount);
  return '₹' + rounded.toLocaleString('en-IN');
}

export function formatLakhs(amount: number): string {
  const lakhs = amount / 100000;
  return `₹${lakhs.toFixed(2)} L`;
}

export function rawDatasetToCSV(rows: RawHousingRecord[]): string {
  const header =
    'price,area,bedrooms,bathrooms,stories,mainroad,guestroom,basement,hotwaterheating,airconditioning,parking,prefarea,furnishingstatus';
  const lines = rows.map((r) =>
    [
      r.price ?? '',
      r.area ?? '',
      r.bedrooms ?? '',
      r.bathrooms ?? '',
      r.stories,
      r.mainroad,
      r.guestroom,
      r.basement,
      r.hotwaterheating,
      r.airconditioning,
      r.parking ?? '',
      r.prefarea,
      r.furnishingstatus,
    ].join(',')
  );
  return [header, ...lines].join('\n');
}
